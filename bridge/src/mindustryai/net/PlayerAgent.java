package mindustryai.net;

import arc.math.geom.Vec2;
import arc.util.Log;
import mindustry.Vars;
import mindustry.content.Blocks;
import mindustry.entities.units.AIController;
import mindustry.entities.units.BuildPlan;
import mindustry.gen.Unit;
import mindustry.type.Item;
import mindustry.world.Block;
import mindustry.world.Tile;
import mindustry.world.blocks.storage.CoreBlock;

/**
 * The agent embodied as a player, not as a god.
 *
 * <p>Until now the bridge edited the world directly: a block appeared instantly, anywhere
 * on the map, for the price of its materials. A human cannot do that. They inhabit a core
 * unit, they have to fly to what they want to build, construction takes time, and mining
 * means holding position over an ore patch until the unit fills up.
 *
 * <p>This controller gives the agent exactly the affordances of a player and nothing more.
 * Every limit here is the engine's own: {@code type.buildRange} decides how far it can
 * reach, {@code type.mineTier} decides which ores it can touch, and the build queue is the
 * same one the game processes for a human. None of it is reimplemented, so none of it can
 * drift from the real rules.
 *
 * <p>The agent expresses intent, not outcome: "go there", "mine that tile", "queue this
 * building". Whether any of it succeeds is up to the simulation.
 */
public class PlayerAgent extends AIController {

    /** Where the unit is trying to go, in world units. Null means hold position. */
    private Vec2 moveTarget;

    /** Tile the unit is trying to mine. Null means stop mining. */
    private Tile mining;

    /** Set when the unit is close enough to the core to unload what it carries. */
    private boolean unloading;

    // Intent -----------------------------------------------------------------------

    public void moveTo(float tileX, float tileY) {
        moveTarget = new Vec2(tileX * Vars.tilesize, tileY * Vars.tilesize);
    }

    public void stopMoving() {
        moveTarget = null;
    }

    /**
     * Queue a building.
     *
     * <p>Range is not checked here on purpose. The engine skips plans that are out of
     * reach and keeps them queued, exactly as it does for a player who queued something
     * across the map, so the agent learns that it has to travel rather than being told no.
     */
    public String build(Block block, int x, int y, int rotation) {
        Unit unit = unit();
        if (unit == null) {
            return "no unit";
        }
        if (block == null) {
            return "unknown block";
        }
        if (!unit.canBuild()) {
            return "this unit cannot build";
        }
        unit.addBuild(new BuildPlan(x, y, rotation, block, null));
        return null;
    }

    public String breakBlock(int x, int y) {
        Unit unit = unit();
        Tile tile = Vars.world.tile(x, y);
        if (unit == null || tile == null) {
            return "no unit or tile";
        }
        unit.addBuild(new BuildPlan(x, y));
        return null;
    }

    public void clearBuildQueue() {
        Unit unit = unit();
        if (unit != null) {
            unit.clearBuilding();
        }
    }

    /**
     * Start mining a tile.
     *
     * <p>Refused for the reasons the game refuses a player: no ore, an ore too hard for
     * this unit's drill tier, or a unit that cannot mine at all. Distance is not one of
     * them: being too far simply means nothing happens until the unit gets closer.
     */
    public String mine(int x, int y) {
        Unit unit = unit();
        Tile tile = Vars.world.tile(x, y);
        if (unit == null || tile == null) {
            return "no unit or tile";
        }
        if (!unit.canMine()) {
            return "this unit cannot mine";
        }

        Item drop = tile.drop();
        if (drop == null) {
            return "nothing to mine there";
        }
        if (!unit.canMine(drop)) {
            return "ore too hard for this unit: " + drop.name;
        }

        mining = tile;
        unit.mineTile = tile;
        return null;
    }

    public void stopMining() {
        mining = null;
        Unit unit = unit();
        if (unit != null) {
            unit.mineTile = null;
        }
    }

    /** Head to the core and hand over whatever is being carried. */
    public void unload() {
        unloading = true;
    }

    // Per-tick behaviour -------------------------------------------------------------

    @Override
    public void updateUnit() {
        Unit unit = unit();
        if (unit == null) {
            return;
        }

        // Building and mining are driven by the engine from the unit's own state, which is
        // why they are not touched here: updateBuilding() walks the plan queue and checks
        // range itself, and mining runs off unit.mineTile.
        if (mining != null && unit.mineTile != mining) {
            unit.mineTile = mining;
        }

        if (unloading) {
            handleUnloading(unit);
            return;
        }

        if (moveTarget != null) {
            approach(unit, moveTarget);
        }

        faceTarget(unit);
    }

    private void handleUnloading(Unit unit) {
        CoreBlock.CoreBuild core = unit.closestCore();
        if (core == null || unit.stack.amount == 0) {
            unloading = false;
            return;
        }

        approach(unit, new Vec2(core.x, core.y));

        // The engine only hands items over by itself when the unit is full and actively
        // mining. An agent that decides to bank a half load has to ask, the same way a
        // player pressing the deposit key does.
        if (unit.within(core, Vars.mineTransferRange)) {
            // Mining keeps refilling the stack while we try to empty it, so mining is
            // stopped for the handover. Otherwise the unit hovers at a few items forever:
            // it deposits and immediately re-mines, and the core never sees a full stack.
            unit.mineTile = null;
            mining = null;

            Item carried = unit.item();
            int amount = unit.stack.amount;
            int accepted = carried == null ? 0 : core.acceptStack(carried, amount, unit);
            int before = carried == null ? -1 : core.items.get(carried);

            if (accepted > 0) {
                core.handleStack(carried, accepted, unit);
                unit.clearItem();
                // The engine plays an item flying to the core here. A headless server
                // draws nothing, so the fact is recorded instead and the viewer animates
                // it, rather than a load of ore vanishing with no explanation.
                lastDeposit = new Deposit(unit.x / Vars.tilesize, unit.y / Vars.tilesize,
                    core.x / Vars.tilesize, core.y / Vars.tilesize, carried.id, accepted);
            }

            Log.info("[mindustry-ai] deposit item=@ amount=@ accepted=@ core=@->@ team=@/@",
                carried, amount, accepted, before,
                carried == null ? -1 : core.items.get(carried),
                unit.team(), core.team);
            unloading = false;
        }
    }

    /** An item handover the viewer has not been told about yet. */
    public record Deposit(float x, float y, float toX, float toY, int item, int amount) {}

    private Deposit lastDeposit;

    /** Hand over the last deposit, once. Null when nothing has been banked since. */
    public Deposit takeDeposit() {
        Deposit deposit = lastDeposit;
        lastDeposit = null;
        return deposit;
    }

    private void approach(Unit unit, Vec2 target) {
        unit.movePref(new Vec2(target).sub(unit.x, unit.y).limit(unit.speed()));
    }

    private void faceTarget(Unit unit) {
        if (mining != null) {
            unit.lookAt(mining.worldx(), mining.worldy());
        } else if (moveTarget != null) {
            unit.lookAt(moveTarget.x, moveTarget.y);
        }
    }

    // State for observations ---------------------------------------------------------

    public boolean isMining() {
        Unit unit = unit();
        return unit != null && unit.mineTile != null;
    }

    public boolean isBuilding() {
        Unit unit = unit();
        return unit != null && unit.activelyBuilding();
    }

    public int queuedPlans() {
        Unit unit = unit();
        return unit == null ? 0 : unit.plans().size;
    }

    public Vec2 target() {
        return moveTarget;
    }

    /**
     * Spawn the unit a player would get from their core, and take control of it.
     *
     * @return the controller, or null if there is no core to spawn from
     */
    public static PlayerAgent spawnAtCore() {
        CoreBlock.CoreBuild core = Vars.state.rules.defaultTeam.core();
        if (core == null) {
            return null;
        }

        var type = ((CoreBlock) core.block).unitType;
        Unit unit = type.create(Vars.state.rules.defaultTeam);
        unit.set(core.x, core.y);
        unit.add();

        PlayerAgent agent = new PlayerAgent();
        unit.controller(agent);
        Log.info("[mindustry-ai] agent embodied as @ at @,@",
            type.name, (int) (core.x / Vars.tilesize), (int) (core.y / Vars.tilesize));
        return agent;
    }
}
