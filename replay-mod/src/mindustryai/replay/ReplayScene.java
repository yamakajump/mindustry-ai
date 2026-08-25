package mindustryai.replay;

import arc.math.Mathf;
import arc.struct.IntMap;
import arc.struct.IntSet;
import mindustry.Vars;
import mindustry.content.Blocks;
import mindustry.entities.units.AIController;
import mindustry.game.Team;
import mindustry.gen.Unit;
import mindustry.type.UnitType;
import mindustry.world.Block;
import mindustry.world.Tile;

/**
 * Puts the recorded world back exactly as it was, rather than simulating towards it.
 *
 * <p>A replay used to be a reconstruction: it replayed the agent's actions on a rebuilt
 * map and let the game work out everything else. That looks reasonable and is not, because
 * everything else is most of what happened. Enemies choose their own paths, turrets choose
 * their own targets, items ride conveyors at their own pace, and none of those choices are
 * in the recording, so the game makes fresh ones and the divergence compounds from the
 * first second. What was shown was a plausible episode, not the episode.
 *
 * <p>Measured on the previous reader: it understood two of the six action types the agent
 * uses, which is 46% of what it did, and 1% of the blocks it built, because a stamp lays
 * thirty-eight at a time and the reader treated it as one placement it did not recognise.
 * Waves were switched off outright, so no enemy ever appeared.
 *
 * <p>So the recording is authoritative here. Units are puppets: they carry no controller
 * that could decide anything, they are placed where the recorder saw them, and between two
 * recorded steps they are interpolated rather than simulated. What the game still does is
 * draw: sprites, shadows, conveyor animation, turret barrels. It renders, it does not
 * decide.
 */
public class ReplayScene {

    /** A recorded unit and the two positions it is travelling between. */
    private static final class Puppet {
        Unit unit;
        float fromX, fromY, toX, toY;
        float fromRot, toRot;
        int health = 100;
    }

    /** Does nothing on purpose, which is the entire point of a puppet. */
    private static final class Frozen extends AIController {
        @Override
        public void updateUnit() {
        }
    }

    private final IntMap<Puppet> puppets = new IntMap<>();
    private final IntSet touched = new IntSet();

    /** The unit the agent was flying, so a viewer can tell it apart from the rest. */
    private int agent = -1;

    public int agent() {
        return agent;
    }

    public Unit agentUnit() {
        Puppet puppet = puppets.get(agent);
        return puppet == null ? null : puppet.unit;
    }

    /** Forget everything, for a seek or a new file. */
    public void clear() {
        for (Puppet puppet : puppets.values()) {
            if (puppet.unit != null) {
                puppet.unit.remove();
            }
        }
        puppets.clear();
        agent = -1;
    }

    /**
     * Apply one recorded step.
     *
     * <p>Buildings first, then units, because a unit standing on a tile that is about to
     * become a wall should be placed after the wall exists rather than be pushed out of it.
     */
    public void apply(ReplayFile.Scene scene) {
        if (scene == null) {
            return;
        }
        if (scene.agent >= 0) {
            agent = scene.agent;
        }

        applyBuildings(scene);
        applyUnits(scene);
        applyDepartures(scene);
    }

    private void applyBuildings(ReplayFile.Scene scene) {
        int width = Vars.world.width();

        for (int i = 0; i + ReplayFile.Scene.BUILDING <= scene.placed.length;
             i += ReplayFile.Scene.BUILDING) {
            int key = scene.placed[i];
            Tile tile = Vars.world.tile(key % width, key / width);
            Block block = Vars.content.block(scene.placed[i + 1]);
            if (tile == null || block == null) {
                continue;
            }
            tile.setBlock(block, Team.get(scene.placed[i + 3]), scene.placed[i + 2]);
            if (tile.build != null) {
                tile.build.health = tile.build.maxHealth() * scene.placed[i + 4] / 100f;
            }
        }

        for (int key : scene.removed) {
            Tile tile = Vars.world.tile(key % width, key / width);
            if (tile != null && tile.build != null) {
                tile.setBlock(Blocks.air);
            }
        }

        for (int i = 0; i + ReplayFile.Scene.HURT <= scene.hurt.length;
             i += ReplayFile.Scene.HURT) {
            int key = scene.hurt[i];
            Tile tile = Vars.world.tile(key % width, key / width);
            if (tile != null && tile.build != null) {
                tile.build.health = tile.build.maxHealth() * scene.hurt[i + 1] / 100f;
            }
        }
    }

    private void applyUnits(ReplayFile.Scene scene) {
        touched.clear();

        for (int index = 0; index < scene.unitCount(); index++) {
            int id = scene.unit(index, 0);
            touched.add(id);

            UnitType type = Vars.content.unit(scene.unit(index, 1));
            if (type == null) {
                continue;
            }
            Team team = Team.get(scene.unit(index, 2));
            float x = scene.unit(index, 3) * Vars.tilesize;
            float y = scene.unit(index, 4) * Vars.tilesize;

            Puppet puppet = puppets.get(id);
            if (puppet == null || puppet.unit == null || !puppet.unit.isAdded()
                || puppet.unit.type != type || puppet.unit.team() != team) {
                if (puppet != null && puppet.unit != null) {
                    puppet.unit.remove();
                }
                puppet = new Puppet();
                puppet.unit = type.create(team);
                puppet.unit.set(x, y);
                puppet.unit.add();
                // No controller means no decisions, which is what makes this a recording.
                puppet.unit.controller(new Frozen());
                puppet.fromX = x;
                puppet.fromY = y;
                puppets.put(id, puppet);
            } else {
                puppet.fromX = puppet.toX;
                puppet.fromY = puppet.toY;
                puppet.fromRot = puppet.toRot;
            }

            puppet.toX = x;
            puppet.toY = y;
            puppet.toRot = scene.unit(index, 5);
            puppet.health = scene.unit(index, 6);
            puppet.unit.health = puppet.unit.maxHealth() * puppet.health / 100f;
        }
    }

    /**
     * Units the recorder stopped seeing.
     *
     * <p>`gone` is the explicit list, and it is not enough on its own: a unit that stops
     * moving stops appearing in the delta, and one that dies between two steps may never
     * be listed at all. Anything not named in this step keeps its last position, which is
     * correct, and only an explicit departure removes it.
     */
    private void applyDepartures(ReplayFile.Scene scene) {
        for (int id : scene.gone) {
            Puppet puppet = puppets.remove(id);
            if (puppet != null && puppet.unit != null) {
                puppet.unit.remove();
            }
        }
    }

    /**
     * Slide every puppet between the step it left and the step it is going to.
     *
     * <p>Recorded steps are thirty ticks apart, which is half a second, so snapping to each
     * one makes a smooth flight look like a slideshow. Interpolating is exact at every
     * recorded instant and honest in between: the unit was somewhere on that line, and no
     * recording could say where without storing every tick.
     *
     * @param alpha how far through the current step, from zero to one
     */
    public void interpolate(float alpha) {
        float t = Mathf.clamp(alpha);
        for (Puppet puppet : puppets.values()) {
            if (puppet.unit == null || !puppet.unit.isAdded()) {
                continue;
            }
            puppet.unit.set(
                Mathf.lerp(puppet.fromX, puppet.toX, t),
                Mathf.lerp(puppet.fromY, puppet.toY, t));
            puppet.unit.rotation = Mathf.slerpDelta(puppet.fromRot, puppet.toRot, 1f);
            // Zeroed every frame: a unit with velocity drifts away from where it was
            // recorded, slowly, and a replay that drifts is the thing being fixed.
            puppet.unit.vel.setZero();
        }
    }
}
