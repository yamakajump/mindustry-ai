package mindustryai.replay;

import arc.Core;
import arc.Events;
import arc.struct.IntMap;
import arc.util.Log;
import arc.util.Time;
import mindustry.Vars;
import mindustry.content.Blocks;
import mindustry.core.GameState.State;
import mindustry.game.EventType.Trigger;
import mindustry.io.SaveIO;
import mindustry.io.SaveOptions;
import mindustry.world.Block;
import mindustry.world.Tile;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;

/**
 * Plays a recorded episode inside a live game, with full transport control.
 *
 * <p>Playback is a state machine driven off the game's own tick, so the world runs exactly
 * as it would in a real match: conveyors animate, items move, waves arrive.
 *
 * <p>Seeking backwards is the hard part. There is no way to run the simulation in reverse,
 * so the controller keeps periodic snapshots of the whole world in memory and rebuilds any
 * earlier moment by restoring the nearest one and replaying forward from it. Snapshots are
 * taken with the engine's own save format, which is the only thing that captures
 * everything: buildings, items in transit, units, wave timers.
 */
public class ReplayController {

    /** Steps between snapshots. Closer is faster to seek, and costs memory per snapshot. */
    private static final int SNAPSHOT_EVERY = 40;

    /** Game ticks each recorded step covers. Must match the recorder. */
    public static final int TICKS_PER_STEP = 30;

    private ReplayFile replay;
    private int cursor;
    private float speed = 1f;
    private float tickBudget;

    /** Set while a seek is rebuilding state, so playback does not fight it. */
    private boolean seeking;

    /** Guards against recursion: extra logic updates fire Trigger.update again. */
    private boolean insideExtraUpdate;

    /** step index to a full world save. */
    private final IntMap<byte[]> snapshots = new IntMap<>();

    private Runnable onChange = () -> {};

    public void install() {
        Events.run(Trigger.update, this::update);

        // Slowing down and speeding up are not symmetric, and treating them the same
        // breaks the game.
        //
        // Below 1x the clock is simply scaled: a smaller delta is always safe, and it
        // slows everything together, conveyors and units included, which is the point.
        //
        // Above 1x the clock is left alone and extra logic updates are run instead.
        // Mindustry clamps delta at 3 for a reason: movement and collision assume small
        // steps, and a delta of 64 would let units pass straight through walls. Fast
        // forward has to be more frames, not bigger ones.
        Time.setDeltaProvider(() -> {
            float raw = Math.min(Core.graphics.getDeltaTime() * 60f, 3f);
            return loaded() && speed < 1f ? raw * speed : raw;
        });
    }

    public void onChange(Runnable listener) {
        this.onChange = listener;
    }

    // Loading ---------------------------------------------------------------------

    public void load(ReplayFile file) {
        this.replay = file;
        this.cursor = 0;
        this.tickBudget = 0f;
        snapshots.clear();
        snapshot(0);
        changed();
    }

    public boolean loaded() {
        return replay != null;
    }

    public ReplayFile file() {
        return replay;
    }

    // Transport -------------------------------------------------------------------

    public void play() {
        if (!loaded()) {
            return;
        }
        if (Vars.state.isGame() && Vars.state.isPaused()) {
            Vars.state.set(State.playing);
        }
        changed();
    }

    public void pause() {
        if (Vars.state.isGame() && !Vars.state.isPaused()) {
            Vars.state.set(State.paused);
        }
        changed();
    }

    public void toggle() {
        if (playing()) {
            pause();
        } else {
            play();
        }
    }

    /**
     * Whether the replay is running.
     *
     * <p>Read from the game state rather than tracked separately. Mindustry already binds
     * space to pause, so a private flag meant one keypress toggled two things that then
     * disagreed, and the button showed the opposite of what was happening.
     */
    public boolean playing() {
        return Vars.state.isGame() && !Vars.state.isPaused();
    }

    public float speed() {
        return speed;
    }

    public void speed(float value) {
        speed = Math.max(0.05f, Math.min(value, 64f));
        changed();
    }

    public int cursor() {
        return cursor;
    }

    public int total() {
        return replay == null ? 0 : replay.frames.size;
    }

    /**
     * Jump to any step, forwards or backwards.
     *
     * <p>Forwards from the current position is just fast replay. Backwards means restoring
     * the nearest earlier snapshot and replaying from there, because a simulation cannot
     * be run in reverse.
     */
    public void seek(int target) {
        if (!loaded()) {
            return;
        }
        target = Math.max(0, Math.min(target, replay.frames.size - 1));

        if (target < cursor) {
            int from = restoreNearest(target);
            if (from < 0) {
                Log.warn("[replay] no snapshot before step @, cannot rewind", target);
                return;
            }
            cursor = from;
        }

        seeking = true;
        while (cursor < target) {
            applyFrame(replay.frames.get(cursor));
            cursor++;
            maybeSnapshot();
        }
        seeking = false;
        tickBudget = 0f;
        changed();
    }

    public void step(int delta) {
        seek(cursor + delta);
    }

    // Snapshots -------------------------------------------------------------------

    private void maybeSnapshot() {
        if (cursor % SNAPSHOT_EVERY == 0) {
            snapshot(cursor);
        }
    }

    private void snapshot(int step) {
        if (snapshots.containsKey(step) || !Vars.state.isGame()) {
            return;
        }
        try {
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            SaveIO.write(out, new SaveOptions());
            snapshots.put(step, out.toByteArray());
        } catch (Throwable t) {
            Log.err("[replay] snapshot at step " + step + " failed", t);
        }
    }

    /** Restore the latest snapshot at or before target, returning its step, or -1. */
    private int restoreNearest(int target) {
        int best = -1;
        for (var entry : snapshots.entries()) {
            if (entry.key <= target && entry.key > best) {
                best = entry.key;
            }
        }
        if (best < 0) {
            return -1;
        }
        try {
            SaveIO.load(new ByteArrayInputStream(snapshots.get(best)), Vars.world.context);
            Vars.state.set(State.paused);
            return best;
        } catch (Throwable t) {
            Log.err("[replay] restoring snapshot " + best + " failed", t);
            return -1;
        }
    }

    // Playback --------------------------------------------------------------------

    private void update() {
        if (seeking || insideExtraUpdate || !loaded() || !Vars.state.isPlaying()) {
            return;
        }

        // Fast forward: run the world several times this frame rather than once with a
        // huge step. Each extra pass is a normal tick, so the simulation stays valid.
        int extra = speed > 1f ? Math.round(speed) - 1 : 0;
        if (extra > 0) {
            insideExtraUpdate = true;
            try {
                for (int i = 0; i < extra; i++) {
                    Vars.logic.update();
                }
            } finally {
                insideExtraUpdate = false;
            }
        }
        if (cursor >= replay.frames.size) {
            pause();
            return;
        }

        // Time.delta already carries the speed multiplier, so actions land at the same
        // point in game time whatever the playback rate.
        tickBudget += Time.delta;
        while (tickBudget >= TICKS_PER_STEP && cursor < replay.frames.size) {
            tickBudget -= TICKS_PER_STEP;
            applyFrame(replay.frames.get(cursor));
            cursor++;
            maybeSnapshot();
            changed();
        }
    }

    private void applyFrame(ReplayFile.Frame frame) {
        if (frame.places()) {
            place(frame);
        } else if (frame.breaks()) {
            Tile tile = Vars.world.tile(frame.x, frame.y);
            if (tile != null && tile.build != null) {
                tile.setBlock(Blocks.air);
            }
        }
    }

    private void place(ReplayFile.Frame frame) {
        Block block = Vars.content.block(frame.block);
        Tile tile = Vars.world.tile(frame.x, frame.y);
        if (block == null || tile == null) {
            return;
        }
        // Recorded actions were already validated when they were played, so they are
        // applied directly. Re-validating here would reject placements that were legal at
        // the time and make the replay diverge from what actually happened.
        tile.setBlock(block, Vars.state.rules.defaultTeam, frame.rotation);
    }

    private void changed() {
        Core.app.post(onChange);
    }
}
