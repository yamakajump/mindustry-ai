package mindustryai.net;

import arc.Events;
import arc.util.serialization.Jval;
import mindustry.Vars;
import mindustry.game.EventType;
import mindustry.type.ItemStack;
import mindustry.world.Block;
import mindustry.world.Build;
import mindustry.world.Tile;

/**
 * Applies agent actions to the world, and reports what is legal.
 *
 * <p>Legality is computed by the engine, never re-derived here. {@code Build.validPlace}
 * knows about terrain, overlap, build radius and team ownership; a reimplementation would
 * drift from the real rules and teach the agent something false.
 *
 * <p><b>Construction is instant.</b> In the real game a placed block becomes a
 * {@code ConstructBlock} that a builder unit has to finish, and with no unit present it
 * would never complete. Resources are still deducted, so the economy is real, but build
 * time and builder range are not modelled yet. Unit control will make this faithful; until
 * then this is a documented simplification, not an oversight.
 */
public class ActionExecutor {

    /** Outcome of one action, reported back in the observation. */
    public record Result(boolean applied, String reason) {
        static Result ok() {
            return new Result(true, null);
        }

        static Result rejected(String reason) {
            return new Result(false, reason);
        }
    }

    /** Place a block, paying for it out of the core. */
    public Result place(String blockName, int x, int y, int rotation) {
        Block block = Vars.content.block(blockName);
        if (block == null) {
            return Result.rejected("no such block: " + blockName);
        }
        if (!block.isPlaceable() || !block.unlockedNow()) {
            return Result.rejected("block is not placeable: " + blockName);
        }
        if (!Build.validPlace(block, Vars.state.rules.defaultTeam, x, y, rotation)) {
            return Result.rejected("invalid placement at " + x + "," + y);
        }
        if (!canAfford(block)) {
            return Result.rejected("cannot afford " + blockName);
        }

        Tile tile = Vars.world.tile(x, y);
        if (tile == null) {
            return Result.rejected("no tile at " + x + "," + y);
        }

        pay(block);
        tile.setBlock(block, Vars.state.rules.defaultTeam, rotation);
        // Placing a block through the world does not announce itself. A player's build
        // finishes through the build queue and fires this, which is what the engine's own
        // counters listen to, so direct mode was silently invisible to every statistic the
        // game keeps: nothing placed, nothing built. Fired here so the two modes are
        // indistinguishable from outside, which is the only reason direct mode is allowed
        // to exist at all.
        Events.fire(new EventType.BlockBuildEndEvent(
            tile, null, Vars.state.rules.defaultTeam, false, null));
        return Result.ok();
    }

    /** Remove a block owned by the agent team. */
    public Result destroy(int x, int y) {
        if (!Build.validBreak(Vars.state.rules.defaultTeam, x, y)) {
            return Result.rejected("nothing breakable at " + x + "," + y);
        }
        Tile tile = Vars.world.tile(x, y);
        if (tile == null) {
            return Result.rejected("no tile at " + x + "," + y);
        }
        tile.setNet(mindustry.content.Blocks.air);
        Events.fire(new EventType.BlockBuildEndEvent(
            tile, null, Vars.state.rules.defaultTeam, true, null));
        return Result.ok();
    }

    private boolean canAfford(Block block) {
        var core = Vars.state.rules.defaultTeam.core();
        if (core == null) {
            return false;
        }
        if (Vars.state.rules.infiniteResources) {
            return true;
        }
        for (ItemStack stack : block.requirements) {
            if (core.items.get(stack.item) < stack.amount) {
                return false;
            }
        }
        return true;
    }

    private void pay(Block block) {
        var core = Vars.state.rules.defaultTeam.core();
        if (core == null || Vars.state.rules.infiniteResources) {
            return;
        }
        for (ItemStack stack : block.requirements) {
            core.items.remove(stack.item, stack.amount);
        }
    }

    /**
     * Which blocks the agent could place right now, by name.
     *
     * <p>Exact: a block appears only if it is unlocked, placeable, and the core holds its
     * full cost. This is the mask for the block head of the action space.
     */
    public Jval affordableBlocks() {
        Jval list = Jval.newArray();
        for (Block block : Vars.content.blocks()) {
            if (block.isPlaceable() && block.unlockedNow() && canAfford(block)) {
                list.asArray().add(Jval.valueOf(block.name));
            }
        }
        return list;
    }
}
