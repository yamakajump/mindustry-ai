package mindustryai;

import arc.mock.MockGraphics;

/**
 * A {@link MockGraphics} that reports a constant frame duration.
 *
 * <p>Mindustry derives simulation speed from {@code Core.graphics.getDeltaTime()} in three
 * separate places: {@code Time.deltaimpl} computes {@code Time.delta} from it, {@code
 * Time.updateGlobal} advances global time with it, and {@code Logic.update} advances
 * {@code state.tick} with it. Replacing the graphics implementation therefore makes all
 * three consistent at once, which is why this is done here rather than through
 * {@code Time.setDeltaProvider}, which would only fix the first.
 *
 * <p>The consequence is a fixed simulation timestep: one frame always advances the world
 * by exactly the same amount of game time, no matter how long the frame actually took.
 * Speed is then controlled purely by how often frames run, see {@link Clock}. This is
 * strictly better than the stock behaviour for our purposes, because a variable timestep
 * makes runs impossible to reproduce and degrades collision handling when frames are slow.
 *
 * <p>Frame counting is delegated to the instance owned by the application loop, since that
 * is the one whose {@code updateTime()} is still being called every frame.
 */
public class FixedStepGraphics extends MockGraphics {
    /** One tick at Mindustry's nominal 60 ticks per second. */
    public static final float DEFAULT_STEP = 1f / 60f;

    private final MockGraphics counting;
    private final float step;

    /**
     * @param counting the instance the application loop still updates, used for frame counts
     * @param step     the fixed frame duration in seconds
     */
    public FixedStepGraphics(MockGraphics counting, float step) {
        this.counting = counting;
        this.step = step;
    }

    @Override
    public float getDeltaTime() {
        return step;
    }

    @Override
    public int getFramesPerSecond() {
        return counting.getFramesPerSecond();
    }

    @Override
    public long getFrameId() {
        return counting.getFrameId();
    }

    /** The fixed timestep in seconds. */
    public float step() {
        return step;
    }
}
