package mindustryai.replay;

import arc.Core;
import arc.Events;
import arc.input.KeyCode;
import arc.files.Fi;
import arc.struct.Seq;
import arc.util.Log;
import mindustry.Vars;
import mindustry.content.Blocks;
import mindustry.game.EventType.ClientLoadEvent;
import mindustry.game.Gamemode;
import mindustry.game.Rules;
import mindustry.game.Team;
import mindustry.maps.Map;
import mindustry.mod.Mod;
import mindustry.type.SectorPreset;
import mindustry.world.Block;
import mindustry.world.Tile;

/**
 * Watch mindustry-ai agents play, inside the game itself.
 *
 * <p>Drop a recorded episode into {@code <data>/replays/}, open the game, and pick it from
 * the replay button. The world runs for real, with a transport bar for play, pause, speed
 * and rewind.
 *
 * <p>This is a client mod rather than a server plugin on purpose. Rendering and interface
 * only exist on the client, and going through a server would mean running one, joining it,
 * and giving up any control finer than "watch it happen".
 */
public class ReplayMod extends Mod {
    private final ReplayController controller = new ReplayController();
    private final ReplayUI ui = new ReplayUI(controller);

    @Override
    public void init() {
        controller.install();

        Events.on(ClientLoadEvent.class, event -> {
            ui.build();
            addMenuButton();
            bindKeys();
            Log.info("[mindustry-ai] replay mod ready, drop files in @",
                Vars.dataDirectory.child("replays").absolutePath());
        });
    }

    /** Replays live in the game's own data directory, so no path juggling is needed. */
    private Fi replayDirectory() {
        Fi directory = Vars.dataDirectory.child("replays");
        if (!directory.exists()) {
            directory.mkdirs();
        }
        return directory;
    }

    private void addMenuButton() {
        Vars.ui.menufrag.addButton("AI replay", mindustry.gen.Icon.play, this::showPicker);
    }

    private void showPicker() {
        Seq<Fi> files = new Seq<>();
        for (Fi file : replayDirectory().list()) {
            if (file.name().endsWith(".jsonl.gz")) {
                files.add(file);
            }
        }

        if (files.isEmpty()) {
            Vars.ui.showInfo("No replays found in\n" + replayDirectory().absolutePath()
                + "\n\nRecord one with the mindustry-ai tooling, then drop the\n"
                + ".jsonl.gz file in that folder.");
            return;
        }

        var dialog = new mindustry.ui.dialogs.BaseDialog("AI replay");
        dialog.cont.pane(list -> {
            for (Fi file : files) {
                list.button(file.name().replace(".jsonl.gz", ""), () -> {
                    dialog.hide();
                    open(file);
                }).width(420f).height(52f).row();
            }
        }).maxHeight(460f);
        dialog.addCloseButton();
        dialog.show();
    }

    private void open(Fi file) {
        try {
            ReplayFile replay = ReplayFile.read(file);
            Log.info("[mindustry-ai] loaded @ with @ steps", file.name(), replay.frames.size);
            startWorld(replay);
        } catch (Throwable t) {
            Log.err("[mindustry-ai] could not read replay", t);
            Vars.ui.showException("Could not read that replay", t);
        }
    }

    /**
     * Load the same world the episode was recorded on.
     *
     * <p>A replay is only actions; the map is what those actions were applied to, so
     * loading the wrong one would place every block in the wrong place.
     */
    private void startWorld(ReplayFile replay) {
        Runnable begin = () -> {
            controller.load(replay);
            controller.pause();
            if (replay.coreX >= 0) {
                Core.camera.position.set(replay.coreX * Vars.tilesize, replay.coreY * Vars.tilesize);
            }
            Vars.ui.showInfoFade("Replay loaded. Space to play, arrows to step.", 5f);
        };

        // The world first, because the replay carries it. Falling back to a named map is
        // only for recordings older than that, and for the campaign presets.
        if (replay.hasWorld()) {
            buildWorld(replay);
            Core.app.post(begin);
            return;
        }

        if (replay.sector != null) {
            SectorPreset preset = Vars.content.sectors().find(s -> s.name.equals(replay.sector));
            if (preset != null) {
                Vars.control.playSector(preset.sector);
                Core.app.post(begin);
                return;
            }
        }

        Map map = Vars.maps.all().find(candidate ->
            candidate.name().equalsIgnoreCase(replay.map)
                || candidate.plainName().equalsIgnoreCase(replay.map.replace('_', ' ')));

        if (map == null) {
            Vars.ui.showInfo(replay.map.isEmpty()
                ? "This replay carries no world and names no map, so there is nothing to "
                  + "play it on. It was recorded before replays became self-contained."
                : "Map not found: " + replay.map);
            return;
        }

        Vars.control.playMap(map, map.applyRules(Gamemode.survival));
        Core.app.post(begin);
    }

    /**
     * Rebuild the exact world the episode was played on, from the replay itself.
     *
     * <p>The agent trains on generated sectors, which have no name for the game to look
     * up: asking for one produced "Map not found:" with nothing after the colon. The
     * recorder already wrote every floor, ore and block into the header, so the world is
     * in the file and there is nothing to look up.
     *
     * <p>Floors and ores go in through the generator. Blocks go in afterwards, one
     * {@code setBlock} at a time, because a multi-tile building has to be linked to the
     * tiles it covers and the exporter writes its id only at its origin. Building them
     * inside the generator would leave a core as a single lonely tile.
     */
    private void buildWorld(ReplayFile replay) {
        Vars.logic.reset();

        Vars.world.loadGenerator(replay.width, replay.height, tiles -> {
            for (int x = 0; x < replay.width; x++) {
                for (int y = 0; y < replay.height; y++) {
                    int index = y * replay.width + x;
                    Block floor = lookup(replay, replay.floor[index], Blocks.stone);
                    Block overlay = lookup(replay, replay.overlay[index], Blocks.air);
                    tiles.set(x, y, new Tile(x, y,
                        floor.isFloor() ? floor : Blocks.stone,
                        overlay.isOverlay() ? overlay : Blocks.air,
                        Blocks.air));
                }
            }
        });

        Vars.state.rules = new Rules();
        Gamemode.survival.apply(Vars.state.rules);
        // Off deliberately. A replay is a recording, not a match: waves arriving on their
        // own timer would fight the agent's recorded actions for the same world.
        Vars.state.rules.waves = false;
        Vars.state.rules.waveTimer = false;

        Vars.logic.play();

        if (replay.block != null) {
            for (int x = 0; x < replay.width; x++) {
                for (int y = 0; y < replay.height; y++) {
                    int index = y * replay.width + x;
                    Block block = lookup(replay, replay.block[index], Blocks.air);
                    if (block == Blocks.air || block.isFloor() || block.isOverlay()) {
                        continue;
                    }
                    Tile tile = Vars.world.tile(x, y);
                    if (tile == null) {
                        continue;
                    }
                    int rotation = replay.rotation == null ? 0 : replay.rotation[index];
                    // Static walls and boulders belong to nobody; anything a team can own
                    // was standing at the start of the episode and belongs to the agent.
                    Team team = block.isStatic() || !block.hasBuilding()
                        ? Team.derelict : Vars.state.rules.defaultTeam;
                    tile.setBlock(block, team, rotation);
                }
            }
        }
    }

    private static Block lookup(ReplayFile replay, int index, Block fallback) {
        String name = replay.palette.get(index);
        if (name == null || name.isEmpty()) {
            return fallback;
        }
        Block block = Vars.content.block(name);
        return block == null ? fallback : block;
    }

    private void bindKeys() {
        Events.run(mindustry.game.EventType.Trigger.update, () -> {
            if (!controller.loaded() || Vars.ui.chatfrag.shown() || Vars.ui.consolefrag.shown()) {
                return;
            }
            // Space is deliberately not bound. Mindustry already pauses on it, and
            // binding it here toggled the game and the controller in opposite directions,
            // so one press appeared to do nothing and the button showed the wrong state.
            if (Core.input.keyTap(KeyCode.right)) {
                controller.step(1);
            }
            if (Core.input.keyTap(KeyCode.left)) {
                controller.step(-1);
            }
        });
    }
}
