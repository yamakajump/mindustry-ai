package mindustryai.net;

import arc.struct.IntMap;
import arc.struct.IntSet;
import arc.util.serialization.Jval;
import mindustry.Vars;
import mindustry.gen.Groups;
import mindustry.gen.Unit;
import mindustry.world.Block;
import mindustry.world.Tile;
import mindustry.world.blocks.ConstructBlock;

/**
 * A frame of everything that moves, so a viewer can animate the match instead of drawing
 * a still picture of it.
 *
 * <p>The observation tensor and the map export both answer a different question. The
 * tensor is what the policy sees: categories, clamped counts, no identity. The map is the
 * terrain, sent once because it barely changes. Neither carries a position between two
 * tiles, the progress on a half-built drill, or a shot in flight, and those are precisely
 * what makes a match look alive rather than like a spreadsheet.
 *
 * <p>Two properties keep the cost low enough to send one of these per agent decision:
 *
 * <ul>
 *   <li><b>Deltas.</b> Buildings are reported when they appear, vanish or take damage,
 *       never as a full list. A developed base is thousands of buildings and would cost
 *       more per frame than the observation tensor it travels beside.
 *   <li><b>Flat arrays.</b> Every record is a run of numbers rather than an object with
 *       named fields. The names would outweigh the data several times over, and a viewer
 *       reads a flat array with an index rather than an allocation per entity.
 * </ul>
 *
 * <p>Item flow on conveyors is deliberately absent. It is the one visible thing that
 * cannot be sent honestly: items move every tick and there are thousands of them, so a
 * frame every thirty ticks would show them teleporting. A viewer that wants them should
 * animate them from the throughput of the conveyor instead.
 */
public class SceneEncoder {

    /** Values per unit record. */
    public static final int UNIT_STRIDE = 9;

    /** Values per placed-building record. */
    public static final int BUILD_STRIDE = 6;

    /** Values per bullet record. */
    public static final int SHOT_STRIDE = 4;

    /**
     * Bullets reported per frame.
     *
     * <p>A late-game defence fires hundreds at once. Past a point they overlap into a
     * single smear on screen, so the cap costs nothing visible and bounds the frame.
     */
    private static final int MAX_SHOTS = 200;

    /** Last state sent per building, keyed by its origin tile, to diff against. */
    private final IntMap<int[]> buildings = new IntMap<>();

    /** Units present in the last frame, so departures can be reported. */
    private final IntSet units = new IntSet();

    /** Units seen while building the current frame. */
    private final IntSet seenUnits = new IntSet();

    /** Origin tiles seen while building the current frame. */
    private final IntSet seenBuildings = new IntSet();

    /** Block ids whose description has already been sent. */
    private final IntSet knownBlocks = new IntSet();

    /** Unit type ids whose name has already been sent. */
    private final IntSet knownTypes = new IntSet();

    private int sequence;

    /**
     * Forget everything, so the next frame is a full one.
     *
     * <p>Called on every map load. Without it the first frame of a new match would be a
     * delta against the previous one and the viewer would keep drawing the old base.
     */
    public void reset() {
        buildings.clear();
        units.clear();
        knownBlocks.clear();
        knownTypes.clear();
        sequence = 0;
    }

    /**
     * Everything that changed since the previous call.
     *
     * @param agentId the unit the agent inhabits, or -1 when it has no body. A viewer
     *     that had to guess which of the units on screen is the agent would guess wrong
     *     whenever two of them are close, and lose the camera every time it did.
     */
    public Jval encode(int agentId) {
        Jval scene = Jval.newObject();
        scene.put("ok", true);
        scene.put("seq", ++sequence);
        scene.put("agent", agentId);

        if (!Vars.state.isGame()) {
            scene.put("playing", false);
            return scene;
        }

        scene.put("playing", true);
        scene.put("tick", Vars.state.tick);
        scene.put("wave", Vars.state.wave);
        scene.put("wave_time", Vars.state.wavetime);
        scene.put("enemies", Vars.state.enemies);
        scene.put("width", Vars.world.width());
        scene.put("height", Vars.world.height());

        Jval newBlocks = Jval.newObject();
        Jval newTypes = Jval.newObject();

        scene.put("units", encodeUnits(newTypes));
        scene.put("gone", takeDepartedUnits());
        encodeBuildings(scene, newBlocks);
        scene.put("shots", encodeShots());

        if (newBlocks.asObject().size > 0) {
            scene.put("blocks", newBlocks);
        }
        if (newTypes.asObject().size > 0) {
            scene.put("types", newTypes);
        }
        return scene;
    }

    // Units ---------------------------------------------------------------------------

    /**
     * Every unit alive, as {@code [id, type, team, x, y, rotation, health, flags, mine]}.
     *
     * <p>Positions are in tiles and keep their fraction: a unit halfway between two tiles
     * is what lets a viewer interpolate rather than snap. Health is a percentage, because
     * a viewer draws a bar and never needs the fourth digit.
     */
    private Jval encodeUnits(Jval newTypes) {
        Jval array = Jval.newArray();
        seenUnits.clear();

        Groups.unit.each(unit -> {
            seenUnits.add(unit.id());
            describeType(unit, newTypes);

            int flags = (unit.mineTile != null ? 1 : 0)
                | (unit.activelyBuilding() ? 2 : 0)
                | (unit.isShooting() ? 4 : 0);

            Tile mining = unit.mineTile;
            int mineTile = mining == null ? -1 : mining.y * Vars.world.width() + mining.x;

            add(array, unit.id());
            add(array, unit.type.id);
            add(array, unit.team().id);
            add(array, round(unit.x / Vars.tilesize));
            add(array, round(unit.y / Vars.tilesize));
            add(array, Math.round(unit.rotation));
            add(array, percent(unit.health(), unit.maxHealth()));
            add(array, flags);
            add(array, mineTile);
        });

        return array;
    }

    private void describeType(Unit unit, Jval newTypes) {
        if (knownTypes.add(unit.type.id)) {
            newTypes.put(String.valueOf(unit.type.id), unit.type.name);
        }
    }

    /** Ids that were in the previous frame and are not in this one. */
    private Jval takeDepartedUnits() {
        Jval gone = Jval.newArray();
        IntSet.IntSetIterator iterator = units.iterator();
        while (iterator.hasNext) {
            int id = iterator.next();
            if (!seenUnits.contains(id)) {
                add(gone, id);
            }
        }

        units.clear();
        units.addAll(seenUnits);
        return gone;
    }

    // Buildings -----------------------------------------------------------------------

    /**
     * Buildings that appeared, changed or vanished.
     *
     * <p>A building under construction reports the block it is becoming and how far along
     * it is, not the scaffold. Reporting the scaffold would be accurate and useless: what
     * someone watching wants to know is what is being built there.
     */
    private void encodeBuildings(Jval scene, Jval newBlocks) {
        Jval placed = Jval.newArray();
        Jval hurt = Jval.newArray();
        seenBuildings.clear();

        Groups.build.each(building -> {
            Tile tile = building.tile;
            if (tile == null || tile.build != building) {
                // Only the origin tile of a multi-tile building, and only once.
                return;
            }

            int key = tile.y * Vars.world.width() + tile.x;
            seenBuildings.add(key);

            Block block = building.block;
            int progress = 100;
            if (building instanceof ConstructBlock.ConstructBuild construct) {
                if (construct.current != null) {
                    block = construct.current;
                }
                progress = clamp(Math.round(construct.progress * 100f));
            }

            int health = percent(building.health(), building.maxHealth());
            int[] current = {block.id, building.rotation, building.team.id, health, progress};
            int[] previous = buildings.get(key);

            boolean appeared = previous == null
                || previous[0] != current[0]
                || previous[1] != current[1]
                || previous[2] != current[2]
                || previous[4] != current[4];

            if (appeared) {
                describeBlock(block, newBlocks);
                add(placed, key);
                add(placed, current[0]);
                add(placed, current[1]);
                add(placed, current[2]);
                add(placed, current[3]);
                add(placed, current[4]);
            } else if (previous[3] != current[3]) {
                // Damage alone is two numbers rather than six: a base under fire changes
                // health on hundreds of buildings a second and nothing else about them.
                add(hurt, key);
                add(hurt, current[3]);
            }

            buildings.put(key, current);
        });

        Jval removed = Jval.newArray();
        IntSet stale = new IntSet();
        for (IntMap.Entry<int[]> entry : buildings) {
            if (!seenBuildings.contains(entry.key)) {
                add(removed, entry.key);
                stale.add(entry.key);
            }
        }
        IntSet.IntSetIterator iterator = stale.iterator();
        while (iterator.hasNext) {
            buildings.remove(iterator.next());
        }

        scene.put("placed", placed);
        scene.put("hurt", hurt);
        scene.put("removed", removed);
    }

    /**
     * Sprite hints for a block the viewer has not seen yet.
     *
     * <p>The map palette only covers what was on the map at load. Everything the agent
     * builds is new to the viewer, and one without the size would paint a 3x3 building
     * nine times over.
     */
    private void describeBlock(Block block, Jval newBlocks) {
        if (!knownBlocks.add(block.id)) {
            return;
        }
        Jval entry = Jval.newObject();
        entry.put("name", block.name);
        entry.put("size", block.size);
        entry.put("rotate", block.rotate);
        newBlocks.put(String.valueOf(block.id), entry);
    }

    // Shots ---------------------------------------------------------------------------

    /** Bullets in flight, as {@code [x, y, rotation, team]}. */
    private Jval encodeShots() {
        Jval array = Jval.newArray();
        int[] count = {0};

        Groups.bullet.each(bullet -> {
            if (count[0] >= MAX_SHOTS) {
                return;
            }
            count[0]++;
            add(array, round(bullet.x / Vars.tilesize));
            add(array, round(bullet.y / Vars.tilesize));
            add(array, Math.round(bullet.rotation()));
            add(array, bullet.team.id);
        });

        return array;
    }

    // Plumbing ------------------------------------------------------------------------

    private static void add(Jval array, int value) {
        array.asArray().add(Jval.valueOf(value));
    }

    private static void add(Jval array, double value) {
        array.asArray().add(Jval.valueOf(value));
    }

    /** Two decimals, because full float precision is noise at this scale. */
    private static double round(float value) {
        return Math.round(value * 100f) / 100.0;
    }

    private static int clamp(int percent) {
        return Math.max(0, Math.min(100, percent));
    }

    private static int percent(float value, float total) {
        if (total <= 0f) {
            return 100;
        }
        return clamp(Math.round(value / total * 100f));
    }
}
