package mindustryai.net;

import arc.struct.IntSet;
import arc.util.serialization.Jval;
import mindustry.Vars;
import mindustry.world.Block;
import mindustry.world.Tile;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Exports the whole map as typed planes, so a viewer can draw it with the game's own
 * sprites instead of coloured squares.
 *
 * <p>The observation tensor deliberately does not carry this. A policy needs categories
 * it can generalise over, and a block identity per tile would be a 400-way categorical
 * channel that means nothing to a convolution. A viewer needs the opposite: exact
 * identities, so it can pick the right sprite.
 *
 * <p>Four planes, concatenated into one binary frame in this order:
 *
 * <ul>
 *   <li>{@code floor}, uint16, the ground block id
 *   <li>{@code overlay}, uint16, the ore or overlay id, 0 when bare
 *   <li>{@code block}, uint16, the building id, 0 when empty
 *   <li>{@code rotation}, uint8, packed with team in the high nibble
 * </ul>
 *
 * <p>Ids are the engine's own, and the accompanying palette maps only the ones actually
 * present on this map. Sending all four hundred block names would be mostly noise.
 */
public class MapExporter {

    /** Bytes per tile across all planes: 2 + 2 + 2 + 1. */
    private static final int BYTES_PER_TILE = 7;

    public byte[] planes() {
        int width = Vars.world.width();
        int height = Vars.world.height();
        int tiles = width * height;

        ByteBuffer buffer = ByteBuffer.allocate(tiles * BYTES_PER_TILE).order(ByteOrder.LITTLE_ENDIAN);

        // Written plane by plane rather than interleaved, so the viewer can take typed
        // array views straight onto the buffer with no per-tile work.
        for (int i = 0; i < tiles; i++) {
            buffer.putShort((short) Vars.world.tiles.geti(i).floorID());
        }
        for (int i = 0; i < tiles; i++) {
            buffer.putShort((short) Vars.world.tiles.geti(i).overlayID());
        }
        for (int i = 0; i < tiles; i++) {
            Tile tile = Vars.world.tiles.geti(i);
            // A multi-tile building fills every tile it covers with its own id. A viewer
            // reading the plane tile by tile then draws a three-by-three core nine times,
            // each one offset by a tile, which stacks into a pile of frames. Only the
            // origin carries the id; the rest of the footprint is left empty.
            boolean origin = tile.build == null || tile.build.tile == tile;
            buffer.putShort((short) (origin ? tile.blockID() : 0));
        }
        for (int i = 0; i < tiles; i++) {
            Tile tile = Vars.world.tiles.geti(i);
            int rotation = tile.build == null ? 0 : tile.build.rotation & 0xF;
            int team = tile.build == null ? 0 : Math.min(tile.build.team.id, 0xF);
            buffer.put((byte) ((team << 4) | rotation));
        }

        return buffer.array();
    }

    /**
     * Names for every block id present on the map, plus the sprite hints a viewer needs.
     *
     * <p>Size matters here: a block is drawn once per tile it covers, so a viewer that
     * did not know a core is 3x3 would paint nine cores.
     */
    public Jval palette() {
        IntSet seen = new IntSet();
        int tiles = Vars.world.width() * Vars.world.height();
        for (int i = 0; i < tiles; i++) {
            Tile tile = Vars.world.tiles.geti(i);
            seen.add(tile.floorID());
            seen.add(tile.overlayID());
            seen.add(tile.blockID());
        }

        Jval palette = Jval.newObject();
        for (Block block : Vars.content.blocks()) {
            if (!seen.contains(block.id)) {
                continue;
            }
            Jval entry = Jval.newObject();
            entry.put("name", block.name);
            entry.put("size", block.size);
            entry.put("solid", block.solid);
            entry.put("rotate", block.rotate);
            if (block instanceof mindustry.world.blocks.environment.Floor floor) {
                entry.put("variants", floor.variants);
                // Which floor wins where two meet, so a viewer blends them in the same
                // direction the engine does. Comparing raw block ids gets it backwards
                // wherever the ids do not happen to follow the blend order.
                entry.put("blend", floor.blendGroup.id);
                entry.put("liquid", floor.isLiquid);
            } else {
                entry.put("variants", 0);
            }
            // What the renderer needs to reproduce two passes it cannot infer from the
            // sprite: which tiles cast the soft shadow the whole map sits under, and
            // which are deep enough inside a rock mass to go black.
            entry.put("shadow", block.hasShadow && block != mindustry.content.Blocks.air);
            entry.put("dark", block.solid && !block.synthetic() && block.fillsTile);
            palette.put(String.valueOf(block.id), entry);
        }
        return palette;
    }

    /** Describes the binary layout so a viewer does not have to hardcode offsets. */
    public Jval layout() {
        int tiles = Vars.world.width() * Vars.world.height();
        Jval layout = Jval.newObject();

        Jval floor = Jval.newObject();
        floor.put("offset", 0);
        floor.put("dtype", "uint16");
        layout.put("floor", floor);

        Jval overlay = Jval.newObject();
        overlay.put("offset", tiles * 2);
        overlay.put("dtype", "uint16");
        layout.put("overlay", overlay);

        Jval block = Jval.newObject();
        block.put("offset", tiles * 4);
        block.put("dtype", "uint16");
        layout.put("block", block);

        Jval rotation = Jval.newObject();
        rotation.put("offset", tiles * 6);
        rotation.put("dtype", "uint8");
        rotation.put("note", "low nibble rotation, high nibble team");
        layout.put("rotation", rotation);

        return layout;
    }
}
