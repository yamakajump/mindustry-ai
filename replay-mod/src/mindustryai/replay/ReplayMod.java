package mindustryai.replay;

import arc.Core;
import arc.Events;
import arc.input.KeyCode;
import arc.files.Fi;
import arc.struct.Seq;
import arc.util.Log;
import mindustry.Vars;
import mindustry.game.EventType.ClientLoadEvent;
import mindustry.game.Gamemode;
import mindustry.maps.Map;
import mindustry.mod.Mod;
import mindustry.type.SectorPreset;

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
                list.button(file.nameWithoutExtension(), () -> {
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
            Vars.ui.showInfo("Map not found: " + replay.map);
            return;
        }

        Vars.control.playMap(map, map.applyRules(Gamemode.survival));
        Core.app.post(begin);
    }

    private void bindKeys() {
        Events.run(mindustry.game.EventType.Trigger.update, () -> {
            if (!controller.loaded() || Vars.ui.chatfrag.shown() || Vars.ui.consolefrag.shown()) {
                return;
            }
            if (Core.input.keyTap(KeyCode.space)) {
                controller.toggle();
            }
            if (Core.input.keyTap(KeyCode.right)) {
                controller.step(1);
            }
            if (Core.input.keyTap(KeyCode.left)) {
                controller.step(-1);
            }
        });
    }
}
