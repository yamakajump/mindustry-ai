package mindustryai.net;

import arc.struct.Seq;
import mindustry.Vars;
import mindustry.gen.Groups;
import mindustry.type.Item;
import mindustry.world.Tile;

/**
 * Encodes the world into a dense spatial tensor.
 *
 * <p>Layout is channel-major, {@code (C, H, W)}, indexed {@code c * H * W + y * W + x}.
 * That is what PyTorch convolutions expect, so the Python side can wrap the buffer without
 * transposing.
 *
 * <p>Values are unsigned bytes rather than floats. Most channels are binary or categorical,
 * only health is genuinely continuous, and a float tensor would be four times the bytes for
 * information the game does not have. Normalisation belongs on the Python side, where the
 * policy can decide what scale it wants.
 *
 * <p>Ore channels are allocated per item that actually drops on the loaded map. A map with
 * no thorium gets no thorium channel, which keeps the tensor honest about what exists.
 */
public class ObservationEncoder {
    /** Channels that are always present, before per-ore channels are appended. */
    public static final String[] BASE_CHANNELS = {
        "solid",         // wall or otherwise impassable
        "buildable",     // floor accepts construction, ignoring what currently sits there
        "block",         // any building present
        "block_ally",    // building belonging to the agent team
        "block_enemy",   // building belonging to anyone else
        "block_health",  // 0 to 255, scaled from health fraction
        "unit_ally",     // count of allied units on the tile, clamped
        "unit_enemy",    // count of hostile units on the tile, clamped
    };

    private final Seq<Item> ores = new Seq<>();
    private String[] channels = BASE_CHANNELS;
    private int width;
    private int height;
    private byte[] buffer = new byte[0];

    /**
     * Recompute the channel layout for the loaded map.
     * Must be called after every map load, before the first encode.
     */
    public void rebuild() {
        width = Vars.world.width();
        height = Vars.world.height();

        ores.clear();
        for (int i = 0; i < width * height; i++) {
            Item drop = Vars.world.tiles.geti(i).drop();
            if (drop != null && !ores.contains(drop)) {
                ores.add(drop);
            }
        }
        ores.sort(item -> item.id);

        channels = new String[BASE_CHANNELS.length + ores.size];
        System.arraycopy(BASE_CHANNELS, 0, channels, 0, BASE_CHANNELS.length);
        for (int i = 0; i < ores.size; i++) {
            channels[BASE_CHANNELS.length + i] = "ore_" + ores.get(i).name;
        }

        buffer = new byte[channels.length * width * height];
    }

    public String[] channels() {
        return channels;
    }

    public int width() {
        return width;
    }

    public int height() {
        return height;
    }

    /** Encode the current world and return the backing buffer, reused between calls. */
    public byte[] encode() {
        if (width != Vars.world.width() || height != Vars.world.height()) {
            rebuild();
        }

        java.util.Arrays.fill(buffer, (byte) 0);

        int plane = width * height;
        int oreBase = BASE_CHANNELS.length;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                Tile tile = Vars.world.tile(x, y);
                if (tile == null) {
                    continue;
                }
                int index = y * width + x;

                if (tile.solid()) {
                    buffer[index] = 1;
                }
                if (!tile.floor().isDeep() && tile.floor().placeableOn) {
                    buffer[plane + index] = 1;
                }

                var building = tile.build;
                if (building != null) {
                    buffer[2 * plane + index] = 1;
                    boolean ally = building.team == Vars.state.rules.defaultTeam;
                    buffer[(ally ? 3 : 4) * plane + index] = 1;
                    float fraction = building.health() / Math.max(1f, building.maxHealth());
                    buffer[5 * plane + index] = (byte) clamp255(Math.round(fraction * 255f));
                }

                Item drop = tile.drop();
                if (drop != null) {
                    int channel = ores.indexOf(drop);
                    if (channel >= 0) {
                        buffer[(oreBase + channel) * plane + index] = 1;
                    }
                }
            }
        }

        // Units are entities, not tiles, so they are accumulated in a second pass.
        Groups.unit.each(unit -> {
            int x = unit.tileX();
            int y = unit.tileY();
            if (x < 0 || y < 0 || x >= width || y >= height) {
                return;
            }
            int index = y * width + x;
            boolean ally = unit.team() == Vars.state.rules.defaultTeam;
            int offset = (ally ? 6 : 7) * plane + index;
            buffer[offset] = (byte) clamp255((buffer[offset] & 0xFF) + 1);
        });

        return buffer;
    }

    private static int clamp255(int value) {
        return value < 0 ? 0 : Math.min(value, 255);
    }
}
