package mindustryai.net;

import arc.ApplicationListener;
import arc.Core;
import arc.util.Log;
import arc.util.serialization.Jval;
import mindustry.Vars;
import mindustry.core.GameState.State;
import mindustry.core.Version;
import mindustry.game.Gamemode;
import mindustry.maps.Map;
import mindustryai.Clock;

/**
 * Drives the game one agent decision at a time.
 *
 * <p>The contract an environment needs is that the world does not move while the agent is
 * choosing. Otherwise the state an action was chosen for is already stale when the action
 * lands, and nothing is reproducible.
 *
 * <p>That is achieved by pausing rather than by blocking. After each observation the game
 * state is set to paused, which makes {@code Logic.update} skip the world entirely, and a
 * step request unpauses it for a fixed number of ticks. No thread is ever held, so the
 * server stays responsive and the console keeps working while an agent thinks.
 *
 * <p>A step spans several ticks on purpose. Deciding 60 times per game second is pointless
 * for a game about building factories, and it would multiply the cost of every observation
 * by an order of magnitude for no gain in control.
 */
public class StepLoop implements ApplicationListener {
    /** Ticks advanced per agent decision when the request does not say. */
    public static final int DEFAULT_REPEAT = 15;

    private final BridgeServer server;
    private final Clock clock;

    private int ticksRemaining;
    private boolean stepping;

    /** The connection a step in progress belongs to. See {@link BridgeServer#session()}. */
    private int steppingSession;

    public StepLoop(BridgeServer server, Clock clock) {
        this.server = server;
        this.clock = clock;
    }

    public void install() {
        Core.app.addListener(this);
    }

    @Override
    public void update() {
        if (stepping) {
            // The agent that asked for this step is gone. Abandon it silently: delivering
            // the reply would hand it to whoever connects next and shift every subsequent
            // exchange by one message.
            if (server.session() != steppingSession) {
                stepping = false;
                freeze();
                return;
            }

            // Only count ticks the world actually ran. Paused frames must not count, or a
            // step would end without the simulation having moved.
            if (!Vars.state.isPaused() && Vars.state.isPlaying()) {
                ticksRemaining--;
            }
            if (ticksRemaining <= 0) {
                stepping = false;
                freeze();
                server.reply(observation(true).toString());
            }
            return;
        }

        String request = server.pollRequest();
        if (request != null) {
            handle(request);
        }
    }

    private void handle(String raw) {
        Jval message;
        try {
            message = Jval.read(raw);
        } catch (Exception e) {
            server.reply(error("malformed json: " + e.getMessage()));
            return;
        }

        Jval command = message.get("cmd");
        if (command == null) {
            server.reply(error("missing 'cmd'"));
            return;
        }

        try {
            switch (command.asString()) {
                case "hello" -> handleHello();
                case "reset" -> handleReset(message);
                case "step" -> handleStep(message);
                case "observe" -> server.reply(observation(false).toString());
                case "close" -> handleClose();
                default -> server.reply(error("unknown command: " + command.asString()));
            }
        } catch (Exception e) {
            Log.err("[mindustry-ai] command failed", e);
            server.reply(error(e.getClass().getSimpleName() + ": " + e.getMessage()));
        }
    }

    private void handleHello() {
        Jval reply = Jval.newObject();
        reply.put("ok", true);
        reply.put("protocol", Protocol.VERSION);
        reply.put("bridge", mindustryai.BridgePlugin.VERSION);
        reply.put("mindustry", Version.build + "." + Version.revision);
        reply.put("clock", clock.isOperational() ? "ok" : "degraded");
        server.reply(reply.toString());
    }

    private void handleReset(Jval message) {
        String mapName = message.get("map") == null ? null : message.get("map").asString();
        String modeName = message.get("mode") == null ? "survival" : message.get("mode").asString();

        Map map = mapName == null
            ? Vars.maps.all().random()
            : Vars.maps.all().find(m -> m.name().equalsIgnoreCase(mapName)
                || m.plainName().replace(' ', '_').equalsIgnoreCase(mapName));

        if (map == null) {
            server.reply(error("no such map: " + mapName));
            return;
        }

        Gamemode mode = Gamemode.valueOf(modeName);

        Vars.world.loadMap(map, map.applyRules(mode));
        Vars.state.rules = map.applyRules(mode);
        Vars.logic.play();
        Vars.netServer.openServer();

        freeze();
        server.reply(observation(false).toString());
    }

    private void handleStep(Jval message) {
        if (!Vars.state.isGame()) {
            server.reply(error("no game in progress, send reset first"));
            return;
        }

        int repeat = message.get("repeat") == null ? DEFAULT_REPEAT : message.get("repeat").asInt();
        if (repeat < 1) {
            server.reply(error("repeat must be at least 1"));
            return;
        }

        ticksRemaining = repeat;
        steppingSession = server.session();
        stepping = true;
        unfreeze();
    }

    private void handleClose() {
        unfreeze();
        Jval reply = Jval.newObject();
        reply.put("ok", true);
        server.reply(reply.toString());
    }

    /** Stop the world so the agent can think against a state that will not move. */
    private void freeze() {
        if (Vars.state.isGame() && !Vars.state.isPaused()) {
            Vars.state.set(State.paused);
        }
    }

    private void unfreeze() {
        if (Vars.state.isGame() && Vars.state.isPaused()) {
            Vars.state.set(State.playing);
        }
    }

    /**
     * Current world state.
     *
     * <p>Scalars only for now. The spatial tensors described in the architecture belong on
     * the binary frame type, and adding them here would mean encoding 90,000 tiles as JSON.
     */
    private Jval observation(boolean stepped) {
        Jval obs = Jval.newObject();
        obs.put("ok", true);
        obs.put("stepped", stepped);

        boolean playing = Vars.state.isGame();
        obs.put("playing", playing);
        obs.put("tick", Vars.state.tick);
        obs.put("wave", Vars.state.wave);
        obs.put("wave_time", Vars.state.wavetime);
        obs.put("enemies", Vars.state.enemies);
        obs.put("game_over", Vars.state.gameOver);

        if (playing && Vars.state.rules != null) {
            var core = Vars.state.rules.defaultTeam.core();
            obs.put("has_core", core != null);
            if (core != null) {
                obs.put("core_health", core.health());
                obs.put("core_x", core.tileX());
                obs.put("core_y", core.tileY());

                Jval items = Jval.newObject();
                for (var item : Vars.content.items()) {
                    int amount = core.items.get(item);
                    if (amount > 0) {
                        items.put(item.name, amount);
                    }
                }
                obs.put("items", items);
            }
            obs.put("map_width", Vars.world.width());
            obs.put("map_height", Vars.world.height());
        }

        return obs;
    }

    private String error(String reason) {
        Jval reply = Jval.newObject();
        reply.put("ok", false);
        reply.put("error", reason);
        return reply.toString();
    }
}
