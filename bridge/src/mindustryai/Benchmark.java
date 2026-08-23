package mindustryai;

import arc.Events;
import arc.util.Log;
import mindustry.game.EventType.Trigger;

import java.util.Locale;

/**
 * Measures how much simulation the server actually produces per wall-clock second.
 *
 * <p>This number governs what is trainable at all, so it is measured rather than assumed.
 *
 * <p>Two counters, because they answer different questions. Frames are application loop
 * iterations and keep counting with no game running. Ticks are game updates, fired only
 * while a match is live and unpaused, and are the figure that matters: an environment step
 * costs game ticks, not frames. A large gap between the two means time is going somewhere
 * other than simulating the world.
 */
public class Benchmark {
    private boolean running;
    private long frames;
    private long ticks;
    private long startNanos;
    private long windowNanos;

    public Benchmark() {
        // Fires every application frame, with or without a game in progress. Used to close
        // the measurement window so that a benchmark started on an idle server still ends.
        Events.run(Trigger.update, this::onFrame);

        // Fires only while a match is live and unpaused. This is real simulation.
        Events.run(Trigger.beforeGameUpdate, this::onTick);
    }

    private void onTick() {
        if (running) {
            ticks++;
        }
    }

    private void onFrame() {
        if (!running) {
            return;
        }
        frames++;

        long elapsed = System.nanoTime() - startNanos;
        if (elapsed < windowNanos) {
            return;
        }

        running = false;
        double seconds = elapsed / 1_000_000_000.0;
        Log.info("bench frames=@ ticks=@ seconds=@ fps=@ tps=@",
            frames,
            ticks,
            fixed(seconds, 3),
            fixed(frames / seconds, 2),
            fixed(ticks / seconds, 2));
    }

    /**
     * Format with a dot decimal separator regardless of the host locale.
     *
     * <p>The default JVM locale here is French, which formats 60.21 as "60,21" and
     * silently breaks every machine-side parser. Machine-readable output must never
     * depend on where the server happens to be running.
     */
    private static String fixed(double value, int decimals) {
        return String.format(Locale.ROOT, "%." + decimals + "f", value);
    }

    /**
     * Open a measurement window. The result is logged as a single line when it closes.
     *
     * @param seconds wall-clock duration of the window
     */
    public void start(int seconds) {
        if (running) {
            Log.warn("bench already running");
            return;
        }
        if (seconds <= 0) {
            Log.err("bench window must be positive");
            return;
        }

        frames = 0;
        ticks = 0;
        windowNanos = seconds * 1_000_000_000L;
        startNanos = System.nanoTime();
        running = true;
        Log.info("bench started window=@s", seconds);
    }
}
