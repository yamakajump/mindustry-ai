package mindustryai.replay;

import arc.Core;
import arc.graphics.Color;
import arc.scene.ui.Slider;
import arc.scene.ui.layout.Table;
import arc.util.Align;
import arc.util.Strings;
import mindustry.Vars;
import mindustry.gen.Icon;
import mindustry.gen.Tex;
import mindustry.ui.Styles;

/**
 * The on-screen transport bar: play, pause, scrub, speed, step, jump to core.
 *
 * <p>Built as an overlay on the running game rather than a separate screen, so the world
 * stays fully visible and interactive underneath. Watching a replay should feel like
 * watching a match with a remote in hand, not like reading a log.
 */
public class ReplayUI {
    private final ReplayController controller;

    private Table root;
    private Slider scrub;
    private boolean scrubbing;

    public ReplayUI(ReplayController controller) {
        this.controller = controller;
    }

    public void build() {
        if (root != null) {
            root.remove();
        }

        root = new Table(Tex.buttonEdge3);
        root.bottom();
        root.margin(8f);
        root.visible(controller::loaded);

        Table transport = new Table();

        transport.button(Icon.left, Styles.emptyi, () -> controller.step(-25))
            .size(44f).tooltip("back 25 steps");
        transport.button(Icon.leftSmall, Styles.emptyi, () -> controller.step(-1))
            .size(44f).tooltip("back one step");

        transport.button(Icon.play, Styles.emptyi, controller::toggle)
            .size(52f).tooltip("play or pause (space)")
            .update(button -> button.getStyle().imageUp = controller.playing() ? Icon.pause : Icon.play);

        transport.button(Icon.rightSmall, Styles.emptyi, () -> controller.step(1))
            .size(44f).tooltip("forward one step");
        transport.button(Icon.right, Styles.emptyi, () -> controller.step(25))
            .size(44f).tooltip("forward 25 steps");

        root.add(transport).row();

        scrub = new Slider(0f, 1f, 1f, false);
        scrub.moved(value -> {
            scrubbing = true;
            controller.seek((int) value);
            scrubbing = false;
        });
        scrub.update(() -> {
            if (!scrubbing) {
                scrub.setRange(0f, Math.max(1, controller.total() - 1));
                scrub.setValue(controller.cursor());
            }
        });
        root.add(scrub).width(520f).height(36f).padTop(4f).row();

        Table status = new Table();
        status.label(() -> Strings.format("step @/@", controller.cursor(), Math.max(0, controller.total() - 1)))
            .color(Color.lightGray).padRight(14f);
        status.label(() -> {
            var frame = currentFrame();
            return frame == null ? "" : Strings.format("wave @   tick @", frame.wave, (int) frame.tick);
        }).color(Color.lightGray).padRight(14f);
        status.label(() -> Strings.format("@x", Strings.autoFixed(controller.speed(), 2)))
            .color(Color.orange);
        root.add(status).padTop(2f).row();

        Table speeds = new Table();
        for (float value : new float[]{0.1f, 0.25f, 0.5f, 1f, 2f, 4f, 8f, 16f, 32f, 64f}) {
            speeds.button(label(value), Styles.flatt, () -> controller.speed(value))
                .width(46f).height(32f).padRight(2f)
                .checked(button -> Math.abs(controller.speed() - value) < 0.01f);
        }
        speeds.button(Icon.eyeSmall, Styles.flati, this::focusCore)
            .size(34f).padLeft(8f).tooltip("jump to the core");
        root.add(speeds).padTop(4f);

        Vars.ui.hudGroup.fill(container -> {
            container.bottom();
            container.add(root).padBottom(70f);
        });

        buildHeader();
    }

    /** A quiet line at the top naming what is being watched. */
    private void buildHeader() {
        Vars.ui.hudGroup.fill(container -> {
            container.top().right();
            container.visible(controller::loaded);
            container.table(Tex.buttonEdge1, panel -> {
                panel.margin(8f);
                panel.label(() -> {
                    var file = controller.file();
                    return file == null ? "" : file.task;
                }).color(Color.orange).align(Align.right).row();
                panel.label(() -> {
                    var file = controller.file();
                    return file == null ? "" : file.note.isEmpty() ? file.map : file.note;
                }).color(Color.lightGray).align(Align.right).row();
                panel.label(() -> {
                    var frame = currentFrame();
                    if (frame == null || frame.items == null) {
                        return "";
                    }
                    StringBuilder text = new StringBuilder();
                    for (var entry : frame.items.asObject()) {
                        text.append(entry.key).append(" ").append(entry.value.asInt()).append("  ");
                    }
                    return text.toString().trim();
                }).color(Color.white).align(Align.right);
            }).padTop(70f).padRight(10f);
        });
    }

    /** Compact speed label: "0.1x" below one, "64x" above. */
    private static String label(float value) {
        return (value < 1f ? Strings.autoFixed(value, 2) : String.valueOf((int) value)) + "x";
    }

    private ReplayFile.Frame currentFrame() {
        var file = controller.file();
        if (file == null || file.frames.isEmpty()) {
            return null;
        }
        return file.frames.get(Math.min(controller.cursor(), file.frames.size - 1));
    }

    private void focusCore() {
        var file = controller.file();
        if (file == null || file.coreX < 0) {
            return;
        }
        Core.camera.position.set(file.coreX * Vars.tilesize, file.coreY * Vars.tilesize);
    }
}
