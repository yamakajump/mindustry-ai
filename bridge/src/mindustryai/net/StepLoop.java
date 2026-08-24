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
    private final ObservationEncoder encoder = new ObservationEncoder();
    private final ActionExecutor actions = new ActionExecutor();
    private final MapExporter exporter = new MapExporter();
    private final SceneEncoder scenes = new SceneEncoder();

    /**
     * The agent's body, when it plays as a player rather than editing the world.
     *
     * <p>Null keeps the old direct-edit behaviour, which stays available because it is
     * far faster to train against and useful as an upper bound: whatever an embodied
     * agent achieves, a disembodied one had fewer excuses.
     */
    private PlayerAgent body;

    /** Outcome of the action carried by the last step, reported in its observation. */
    private ActionExecutor.Result lastAction;

    /** Whether the agent asked for spatial tensors. Off by default: they are large. */
    private boolean sendTensor;

    /**
     * Frames a step may span before it is abandoned.
     *
     * <p>A step only completes when the world actually runs, so anything that stops it
     * running (a game over, a state the bridge did not anticipate) would otherwise leave
     * {@code stepping} true forever. While stepping, requests are not consumed, so a stuck
     * step does not fail one call: it wedges the whole connection, and the agent sees an
     * unexplained timeout. Bounding it turns that into an error message.
     */
    private static final int STEP_FRAME_BUDGET = 60 * 60;

    private int ticksRemaining;
    private boolean stepping;
    private int framesSpentStepping;

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
                respond(observation(true));
                return;
            }

            if (++framesSpentStepping > STEP_FRAME_BUDGET) {
                stepping = false;
                freeze();
                Log.warn("[mindustry-ai] step abandoned after @ frames with @ ticks left, "
                    + "playing=@ paused=@ gameOver=@",
                    framesSpentStepping, ticksRemaining,
                    Vars.state.isPlaying(), Vars.state.isPaused(), Vars.state.gameOver);
                server.reply(error("step did not complete: the world stopped advancing"
                    + " (playing=" + Vars.state.isPlaying()
                    + " paused=" + Vars.state.isPaused()
                    + " gameOver=" + Vars.state.gameOver + ")"));
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
                case "hello" -> handleHello(message);
                case "reset" -> handleReset(message);
                case "step" -> handleStep(message);
                case "act" -> handleAct(message);
                case "blocks" -> handleBlocks();
                case "embody" -> handleEmbody();
                case "map" -> handleMap();
                case "sector" -> handleSector(message);
                case "sectors" -> handleSectors();
                case "observe" -> respond(observation(false));
                case "scene" -> handleScene();
                case "close" -> handleClose();
                default -> server.reply(error("unknown command: " + command.asString()));
            }
        } catch (Exception e) {
            Log.err("[mindustry-ai] command failed", e);
            server.reply(error(e.getClass().getSimpleName() + ": " + e.getMessage()));
        }
    }

    private void handleHello(Jval message) {
        if (message.get("tensor") != null) {
            sendTensor = message.get("tensor").asBool();
        }
        Jval reply = Jval.newObject();
        reply.put("ok", true);
        reply.put("protocol", Protocol.VERSION);
        reply.put("bridge", mindustryai.BridgePlugin.VERSION);
        reply.put("mindustry", Version.build + "." + Version.revision);
        reply.put("clock", clock.isOperational() ? "ok" : "degraded");
        reply.put("tensor", sendTensor);

        Jval names = Jval.newArray();
        for (String channel : encoder.channels()) {
            names.asArray().add(Jval.valueOf(channel));
        }
        reply.put("channels", names);
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

        // The engine keeps team data across a world load: cores, buildings and plans from
        // the previous match survive into the next one. The core of the old map then
        // answers every question about where the base is, on a map where it does not
        // exist. The game resets before it loads, and so does this.
        Vars.logic.reset();
        Vars.world.loadMap(map, map.applyRules(mode));
        Vars.state.rules = map.applyRules(mode);
        Vars.logic.play();
        if (!Vars.net.server()) {
            Vars.netServer.openServer();
        }

        encoder.rebuild();
        scenes.reset();
        freeze();
        respond(observation(false));
    }

    /** Take a body, so the agent plays under the same limits a human has. */
    private void handleEmbody() {
        if (!Vars.state.isGame()) {
            server.reply(error("no game in progress, send reset first"));
            return;
        }
        body = PlayerAgent.spawnAtCore();
        if (body == null) {
            server.reply(error("no core to spawn from"));
            return;
        }
        respond(observation(false));
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

        lastAction = message.get("action") == null ? null : apply(message.get("action"));

        ticksRemaining = repeat;
        framesSpentStepping = 0;
        steppingSession = server.session();
        stepping = true;
        unfreeze();
    }

    private void handleAct(Jval message) {
        if (!Vars.state.isGame()) {
            server.reply(error("no game in progress, send reset first"));
            return;
        }
        lastAction = apply(message.get("action"));
        respond(observation(false));
    }

    private void handleBlocks() {
        Jval reply = Jval.newObject();
        reply.put("ok", true);
        reply.put("affordable", actions.affordableBlocks());
        server.reply(reply.toString());
    }

    /**
     * Everything that moved since the last call: units, buildings, shots.
     *
     * <p>Separate from the observation because the two have different audiences and
     * different costs. The observation is what the policy consumes on every step of
     * every environment; this is what a person watching one match consumes, and asking
     * for it on a match nobody is looking at would be pure waste.
     */
    private void handleScene() {
        int agent = body != null && body.unit() != null ? body.unit().id() : -1;
        server.reply(scenes.encode(agent, body == null ? null : body.takeDeposit()).toString());
    }

    /**
     * Send the full typed map: floors, overlays, buildings, rotations, plus a palette.
     *
     * <p>This is what lets a viewer draw the game with its own sprites rather than
     * coloured squares. It is a separate command because it is large and rarely changes:
     * once per map load, not once per step.
     */
    private void handleMap() {
        if (!Vars.state.isGame()) {
            server.reply(error("no map loaded, send reset first"));
            return;
        }

        Jval reply = Jval.newObject();
        reply.put("ok", true);
        reply.put("width", Vars.world.width());
        reply.put("height", Vars.world.height());
        reply.put("palette", exporter.palette());
        reply.put("layout", exporter.layout());

        byte[] planes = exporter.planes();
        Jval spec = Jval.newObject();
        spec.put("bytes", planes.length);
        spec.put("dtype", "mixed");
        Jval shape = Jval.newArray();
        shape.asArray().add(Jval.valueOf(planes.length));
        spec.put("shape", shape);
        reply.put("tensor", spec);

        server.reply(reply.toString(), planes);
    }

    /**
     * Load a campaign sector rather than a custom map.
     *
     * <p>Ground Zero is the first sector of the Serpulo campaign, and capturing it is a
     * real objective the game itself defines: survive to wave 10. That makes a far more
     * meaningful benchmark than a number invented for a custom map.
     */
    private void handleSector(Jval message) {
        mindustry.type.Sector sector;

        if (message.get("index") != null) {
            // A procedural sector, generated by the planet's own generator. Serpulo has
            // several hundred of them, which is what turns "learn this map" into "learn
            // the game": an agent that meets a new world every episode cannot memorise
            // where the copper is.
            mindustry.type.Planet planet = serpulo();
            int index = message.get("index").asInt();
            if (planet == null || index < 0 || index >= planet.sectors.size) {
                server.reply(error("no sector at index " + index));
                return;
            }
            sector = planet.sectors.get(index);
            // Enemy bases are stamped from a registry of prefabs that a headless server
            // never loads, and the generator walks straight into a null item. The sector
            // still has its wave spawns, which is where the pressure comes from anyway.
            sector.generateEnemyBase = false;
        } else {
            String name = message.get("name") == null ? "groundZero" : message.get("name").asString();
            mindustry.type.SectorPreset preset = Vars.content.getByName(
                mindustry.ctype.ContentType.sector, name);
            if (preset == null) {
                server.reply(error("no such sector: " + name));
                return;
            }
            sector = preset.sector;
        }

        // The engine keeps team data across a world load: cores, buildings and plans from
        // the previous match survive into the next one. The core of the old map then
        // answers every question about where the base is, on a map where it does not
        // exist. The game resets before it loads, and so does this.
        Vars.logic.reset();
        Vars.world.loadSector(sector);
        Vars.state.rules.sector = sector;
        Vars.logic.play();
        if (!Vars.net.server()) {
            Vars.netServer.openServer();
        }

        prepareCampaign();
        if (!ensureCore()) {
            server.reply(error("no room for a core on this sector"));
            return;
        }
        applyLoadout(message.get("loadout"));

        encoder.rebuild();
        scenes.reset();
        freeze();
        respond(observation(false));
    }

    /**
     * Give the team a core, because a generated sector does not come with one.
     *
     * <p>In the campaign you land on a sector and the launch places your loadout. Loading
     * one directly skips all of that and hands back a world with terrain, spawns and no
     * way to play it. A core shard on the first clear ground near the middle is the
     * smallest thing that makes the sector a game, and it is what the smallest launch
     * loadout would have given.
     *
     * @return false when the sector has no room at all, which is a sector to skip
     */
    private boolean ensureCore() {
        if (Vars.state.rules.defaultTeam.core() != null) {
            return true;
        }

        int width = Vars.world.width(), height = Vars.world.height();
        int cx = width / 2, cy = height / 2;
        int reach = Math.max(width, height) / 2;

        // Outward from the middle, because the middle is furthest from the wave spawns
        // that sit on the rim.
        for (int radius = 0; radius < reach; radius += 2) {
            for (int dx = -radius; dx <= radius; dx += 2) {
                for (int dy = -radius; dy <= radius; dy += 2) {
                    if (Math.max(Math.abs(dx), Math.abs(dy)) != radius && radius > 0) {
                        continue;
                    }
                    int x = cx + dx, y = cy + dy;
                    if (clearFor(x, y, 3)) {
                        Vars.world.tile(x, y).setBlock(
                            mindustry.content.Blocks.coreShard, Vars.state.rules.defaultTeam);
                        Log.info("[mindustry-ai] placed a core at @,@", x, y);
                        return Vars.state.rules.defaultTeam.core() != null;
                    }
                }
            }
        }
        return false;
    }

    /** Whether a square of the given size centred here is open ground. */
    private boolean clearFor(int x, int y, int size) {
        int half = size / 2 + 1;
        for (int ox = -half; ox <= half; ox++) {
            for (int oy = -half; oy <= half; oy++) {
                var tile = Vars.world.tile(x + ox, y + oy);
                if (tile == null || tile.solid() || tile.floor().isLiquid || tile.block() != mindustry.content.Blocks.air) {
                    return false;
                }
            }
        }
        return true;
    }

    /** Serpulo, the planet the whole curriculum lives on. */
    private mindustry.type.Planet serpulo() {
        return Vars.content.planets().find(planet -> planet.name.equals("serpulo"));
    }

    /**
     * Every sector a training run may draw from, with the numbers needed to choose.
     *
     * <p>Only the procedural ones: a preset sector is a hand-made map with a script, and
     * there are a dozen of them against several hundred generated. Threat travels with
     * each, because a curriculum that widens from calm sectors to dangerous ones is the
     * cheapest difficulty ordering available and the game already computed it.
     */
    private void handleSectors() {
        mindustry.type.Planet planet = serpulo();
        if (planet == null) {
            server.reply(error("serpulo is missing"));
            return;
        }

        Jval indices = Jval.newArray();
        Jval threats = Jval.newArray();
        for (int i = 0; i < planet.sectors.size; i++) {
            mindustry.type.Sector sector = planet.sectors.get(i);
            if (sector.preset != null) {
                continue;
            }
            indices.asArray().add(Jval.valueOf(i));
            threats.asArray().add(Jval.valueOf(Math.round(sector.threat * 100) / 100.0));
        }

        Jval reply = Jval.newObject();
        reply.put("ok", true);
        reply.put("planet", planet.name);
        reply.put("total", planet.sectors.size);
        reply.put("sectors", indices);
        reply.put("threats", threats);
        server.reply(reply.toString());
    }

    /**
     * Give a directly loaded sector the starting conditions a campaign player would have.
     *
     * <p>Two things are missing otherwise, and both are silent.
     *
     * <p>Nothing is researched, so {@code unlockedNow()} is false for every block and the
     * agent cannot place a single one. Measured on Ground Zero: 39 attempts, 39 refusals,
     * all reading "block is not placeable". A player starting the campaign has the basic
     * blocks available, so unlocking them here reproduces the real starting point rather
     * than granting an advantage.
     *
     * <p>And {@code Rules.waves} defaults to false, so no wave ever arrives. Ground Zero
     * is captured by surviving to wave 10, which is unreachable if waves never start: the
     * episode ran 120,000 ticks and stayed on wave 1.
     */
    private void prepareCampaign() {
        Vars.state.rules.waves = true;
        Vars.state.rules.waveTimer = true;

        // Ground Zero is the campaign's tutorial sector, and its map carries the tutorial
        // script: build a drill, open the tech tree, research this. An agent is not doing
        // a tutorial, and leaving the script running puts objectives on screen for anyone
        // watching that have nothing to do with what the run is measuring.
        if (Vars.state.rules.objectives != null) {
            Vars.state.rules.objectives.clear();
        }
        Vars.state.rules.objectiveFlags.clear();

        // Fog hides everything the team has not explored, which is a mechanic for a human
        // discovering a map and pure noise for a training run: it makes a spectator think
        // the agent is building in the void, and it hides the map from anyone watching.
        Vars.state.rules.fog = false;
        Vars.state.rules.staticFog = false;

        Log.info("[mindustry-ai] sector prepared: limitMapArea=@ fog=@ waves=@ size=@x@",
            Vars.state.rules.limitMapArea, Vars.state.rules.fog, Vars.state.rules.waves,
            Vars.world.width(), Vars.world.height());

        for (var block : Vars.content.blocks()) {
            if (block.isPlaceable()) {
                block.unlock();
                Vars.state.rules.researched.add(block);
            }
        }
        for (var item : Vars.content.items()) {
            Vars.state.rules.researched.add(item);
        }
    }

    /**
     * Stock the core so the agent can actually build.
     *
     * <p>A campaign sector normally receives its starting items from the loadout the
     * player launched with, through a chain of conditions in {@code Logic.play} that does
     * not fire for a sector loaded directly. Measured on Ground Zero, the core came up
     * empty, which leaves an agent unable to place a single block. Filling it here is
     * explicit and deterministic rather than dependent on that chain.
     *
     * @param requested optional map of item name to amount, defaulting to the rules loadout
     */
    private void applyLoadout(Jval requested) {
        var core = Vars.state.rules.defaultTeam.core();
        if (core == null) {
            return;
        }

        if (requested != null) {
            for (var entry : requested.asObject()) {
                var item = Vars.content.item(entry.key);
                if (item != null) {
                    core.items.set(item, Math.min(entry.value.asInt(), core.storageCapacity));
                }
            }
            return;
        }

        boolean empty = true;
        for (var item : Vars.content.items()) {
            if (core.items.get(item) > 0) {
                empty = false;
                break;
            }
        }
        if (!empty) {
            return;
        }

        int given = 0;
        for (var stack : Vars.state.rules.loadout) {
            // A generated sector carries a loadout with null entries in it, and adding one
            // throws inside the engine's item module with nothing to say which stack was
            // at fault. Skipped rather than trusted.
            if (stack != null && stack.item != null) {
                core.items.add(stack.item, Math.min(stack.amount, core.storageCapacity));
                given += stack.amount;
            }
        }

        if (given == 0) {
            // Nothing usable in the rules, which is the normal case for a sector loaded
            // outside a launch. An empty core is an unplayable sector, so it gets what the
            // smallest launch loadout would have carried.
            core.items.add(mindustry.content.Items.copper, Math.min(300, core.storageCapacity));
            core.items.add(mindustry.content.Items.lead, Math.min(300, core.storageCapacity));
        }
    }

    /** Decode and apply one action. Never throws: an illegal action is data, not a fault. */
    private ActionExecutor.Result apply(Jval action) {
        if (action == null) {
            return null;
        }
        Jval kind = action.get("type");
        if (kind == null) {
            return new ActionExecutor.Result(false, "action is missing 'type'");
        }

        try {
            return switch (kind.asString()) {
                case "noop" -> new ActionExecutor.Result(true, null);
                case "move" -> embodied(() -> {
                    body.moveTo(action.get("x").asFloat(), action.get("y").asFloat());
                    return null;
                });
                case "build" -> embodied(() -> body.build(
                    Vars.content.block(action.get("block").asString()),
                    action.get("x").asInt(),
                    action.get("y").asInt(),
                    action.get("rotation") == null ? 0 : action.get("rotation").asInt()));
                case "demolish" -> embodied(() -> body.breakBlock(
                    action.get("x").asInt(), action.get("y").asInt()));
                case "mine" -> embodied(() -> body.mine(
                    action.get("x").asInt(), action.get("y").asInt()));
                case "stop_mine" -> embodied(() -> {
                    body.stopMining();
                    return null;
                });
                case "unload" -> embodied(() -> {
                    body.unload();
                    return null;
                });
                case "stop" -> embodied(() -> {
                    body.stopMoving();
                    return null;
                });
                case "place" -> actions.place(
                    action.get("block").asString(),
                    action.get("x").asInt(),
                    action.get("y").asInt(),
                    action.get("rotation") == null ? 0 : action.get("rotation").asInt());
                case "break" -> actions.destroy(
                    action.get("x").asInt(),
                    action.get("y").asInt());
                default -> new ActionExecutor.Result(false, "unknown action: " + kind.asString());
            };
        } catch (Exception e) {
            return new ActionExecutor.Result(false, e.getClass().getSimpleName() + ": " + e.getMessage());
        }
    }

    /** Run a player action, or report that the agent has no body yet. */
    private ActionExecutor.Result embodied(java.util.function.Supplier<String> action) {
        if (body == null) {
            return new ActionExecutor.Result(false, "agent has no body, send embody first");
        }
        String problem = action.get();
        return problem == null
            ? new ActionExecutor.Result(true, null)
            : new ActionExecutor.Result(false, problem);
    }

    private void handleClose() {
        // Deliberately leaves the world frozen. Resuming it would let the map run on
        // between episodes, burning CPU and drifting the state a later reset inherits.
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
        // What the team has standing. The reward uses it as the measure of capability: a
        // pile of ore is a stock, a factory is a machine that keeps making one.
        obs.put("built", playing && Vars.state.rules != null
            ? Vars.state.rules.defaultTeam.data().buildings.size : 0);

        if (lastAction != null) {
            Jval outcome = Jval.newObject();
            outcome.put("applied", lastAction.applied());
            if (lastAction.reason() != null) {
                outcome.put("reason", lastAction.reason());
            }
            obs.put("action", outcome);
        }

        // Every field below is always present, even with no core. An observation whose
        // keys come and go forces every consumer to guess, and a policy fed a vector that
        // silently changes shape learns nothing useful from it.
        Jval items = Jval.newObject();
        obs.put("has_core", false);
        obs.put("core_health", 0);
        obs.put("core_max_health", 0);
        obs.put("core_x", -1);
        obs.put("core_y", -1);
        obs.put("map_width", Vars.world.width());
        obs.put("map_height", Vars.world.height());

        if (playing && Vars.state.rules != null) {
            var core = Vars.state.rules.defaultTeam.core();
            obs.put("has_core", core != null);
            if (core != null) {
                obs.put("core_health", core.health());
                obs.put("core_max_health", core.maxHealth());
                obs.put("core_x", core.tileX());
                obs.put("core_y", core.tileY());

                for (var item : Vars.content.items()) {
                    int amount = core.items.get(item);
                    if (amount > 0) {
                        items.put(item.name, amount);
                    }
                }
            }
        }
        obs.put("items", items);

        if (body != null && body.unit() != null) {
            var unit = body.unit();
            Jval self = Jval.newObject();
            self.put("x", unit.x / Vars.tilesize);
            self.put("y", unit.y / Vars.tilesize);
            self.put("health", unit.health());
            self.put("mining", body.isMining());
            self.put("building", body.isBuilding());
            self.put("plans", body.queuedPlans());
            // In tiles, because every coordinate the agent sees is in tiles.
            self.put("build_range", unit.type.buildRange / Vars.tilesize);
            self.put("mine_range", unit.type.mineRange / Vars.tilesize);
            self.put("mine_tier", unit.type.mineTier);
            self.put("carrying", unit.stack.amount);
            self.put("carrying_item", unit.stack.item == null ? "" : unit.stack.item.name);
            self.put("capacity", unit.type.itemCapacity);
            obs.put("unit", self);
        }
        obs.put("embodied", body != null);

        return obs;
    }

    /**
     * Send an observation, with the spatial tensor attached when the agent asked for it.
     *
     * <p>The JSON frame always carries the tensor's shape and dtype, so the client knows
     * how to read the binary frame that follows without hardcoding the layout.
     */
    private void respond(Jval obs) {
        if (!sendTensor || !Vars.state.isGame()) {
            server.reply(obs.toString());
            return;
        }

        byte[] tensor = encoder.encode();

        Jval shape = Jval.newArray();
        shape.asArray().add(Jval.valueOf(encoder.channels().length));
        shape.asArray().add(Jval.valueOf(encoder.height()));
        shape.asArray().add(Jval.valueOf(encoder.width()));

        // Channel names travel with every tensor, not just with the handshake. Ore
        // channels only exist once a map is loaded, so a list captured at hello time
        // describes a layout that no longer matches what is being sent.
        Jval names = Jval.newArray();
        for (String channel : encoder.channels()) {
            names.asArray().add(Jval.valueOf(channel));
        }

        Jval spec = Jval.newObject();
        spec.put("shape", shape);
        spec.put("dtype", "uint8");
        spec.put("bytes", tensor.length);
        spec.put("channels", names);
        obs.put("tensor", spec);

        server.reply(obs.toString(), tensor);
    }

    private String error(String reason) {
        Jval reply = Jval.newObject();
        reply.put("ok", false);
        reply.put("error", reason);
        return reply.toString();
    }
}
