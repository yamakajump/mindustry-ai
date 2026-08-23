package mindustryai;

import arc.Core;
import arc.mock.MockGraphics;
import arc.util.Log;

import java.lang.reflect.Field;

/**
 * Controls how fast the simulation advances relative to wall-clock time.
 *
 * <p>Two independent mechanisms, which is what makes this work:
 *
 * <ul>
 *   <li>A fixed timestep, installed by swapping {@code Core.graphics} for a
 *       {@link FixedStepGraphics}. Every frame advances the world by the same amount of
 *       game time regardless of how long it really took.
 *   <li>A frame budget, set by writing {@code HeadlessApplication.renderInterval}. The
 *       headless loop sleeps until that interval has elapsed, so shrinking it runs more
 *       frames per second, and zero removes the sleep entirely.
 * </ul>
 *
 * <p>Together they give clean acceleration: the world advances at a constant, reproducible
 * rate per frame, and the frames simply arrive faster. Deliberately not done by inflating
 * {@code Time.delta}, which would make units move further per step and let them tunnel
 * through walls, nor by running extra logic updates per frame, which would desynchronise
 * anything driven by frame counts.
 *
 * <p>Reflection is required because {@code renderInterval} has no setter. The field is
 * read fresh on every loop iteration, so writes take effect immediately.
 */
public class Clock {
    /** Mindustry's nominal simulation rate. */
    public static final int TICKS_PER_SECOND = 60;

    private static final long REALTIME_INTERVAL_NANOS = 1_000_000_000L / TICKS_PER_SECOND;

    private Field renderIntervalField;
    private boolean fixedStepInstalled;
    private int speed = 1;

    /**
     * Install the fixed timestep and locate the frame budget field.
     *
     * @return true if the clock is fully operational
     */
    public boolean install() {
        if (Core.graphics instanceof MockGraphics mock && !(Core.graphics instanceof FixedStepGraphics)) {
            Core.graphics = new FixedStepGraphics(mock, FixedStepGraphics.DEFAULT_STEP);
            fixedStepInstalled = true;
            Log.info("[mindustry-ai] fixed timestep installed step=@s", FixedStepGraphics.DEFAULT_STEP);
        }

        try {
            renderIntervalField = Core.app.getClass().getDeclaredField("renderInterval");
            renderIntervalField.setAccessible(true);
        } catch (NoSuchFieldException | SecurityException e) {
            Log.err("[mindustry-ai] cannot reach renderInterval on @: @",
                Core.app.getClass().getName(), e.toString());
            renderIntervalField = null;
        }

        return isOperational();
    }

    /** Whether both the fixed timestep and the frame budget are under our control. */
    public boolean isOperational() {
        return fixedStepInstalled && renderIntervalField != null;
    }

    /**
     * Set the simulation speed.
     *
     * @param multiplier 1 for realtime, higher for faster, {@link Integer#MAX_VALUE} for uncapped
     */
    public void setSpeed(int multiplier) {
        int requested = Math.max(1, multiplier);
        if (renderIntervalField == null) {
            Log.err("[mindustry-ai] speed unavailable, renderInterval was never located");
            return;
        }

        // Zero means the loop never sleeps. Anything else is the per-frame budget.
        long interval = requested >= TICKS_PER_SECOND * 1000 ? 0L : REALTIME_INTERVAL_NANOS / requested;

        try {
            renderIntervalField.setLong(Core.app, interval);
            speed = requested;
            Log.info("speed set multiplier=@ interval=@ns", requested, interval);
        } catch (IllegalAccessException e) {
            Log.err("[mindustry-ai] failed to set renderInterval: @", e.toString());
        }
    }

    /** The multiplier currently requested. Not a measurement of what is achieved. */
    public int speed() {
        return speed;
    }
}
