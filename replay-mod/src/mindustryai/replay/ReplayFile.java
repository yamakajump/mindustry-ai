package mindustryai.replay;

import arc.files.Fi;
import arc.struct.IntMap;
import arc.struct.Seq;
import arc.util.serialization.Jval;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.zip.GZIPInputStream;
import java.util.zip.Inflater;

/**
 * A recorded episode, read from the gzipped JSON Lines the Python recorder writes.
 *
 * <p>The mod parses the file itself rather than receiving it over a socket, so watching a
 * replay needs nothing running but Mindustry.
 */
public class ReplayFile {

    /** One agent decision and the state it produced. */
    public static class Frame {
        public int step;
        public float tick;
        public int wave;
        public float reward;

        /** Null when the agent did nothing this step. */
        public String actionType;
        public String block;
        public int x, y, rotation;

        public boolean refused;
        public Jval items;

        public boolean places() {
            return "place".equals(actionType);
        }

        public boolean breaks() {
            return "break".equals(actionType);
        }

        public boolean stamps() {
            return "stamp".equals(actionType);
        }

        /**
         * Blocks a stamp laid, four numbers each: x, y, block name index, rotation.
         *
         * <p>A stamp is one action that places a whole structure, so a reader that treats
         * it as a single placement draws almost nothing. Measured on a training archive:
         * 5,772 stamps carrying 425,958 blocks against 5,846 placed one at a time, so 99%
         * of everything built arrived through an action the reader did not understand.
         */
        public int[] cells = EMPTY;
        public String[] cellBlocks = NO_STRINGS;

        /** What moved this step. Null on a replay written before scenes were recorded. */
        public Scene scene;
    }

    /**
     * Everything that moved during one step, exactly as the recorder saw it.
     *
     * <p>This is what separates a recording from a reconstruction. Replaying the agent's
     * actions on a rebuilt world re-simulates everything else, and enemies, turret aim and
     * items in transit all diverge from the episode being claimed. Here they are stated.
     *
     * <p>Every array is a flat run of fixed-width entries, which is how the bridge sends
     * them: it costs a fraction of the equivalent objects and the widths never change.
     */
    public static class Scene {
        /** id, type, team, x, y, rotation, health percent, flags, mined tile, item, amount. */
        public static final int UNIT = 11;
        /** tile, block, rotation, team, health percent, build progress. */
        public static final int BUILDING = 6;
        /** tile, health percent. Damage alone, because a base under fire changes nothing else. */
        public static final int HURT = 2;
        /** x, y, rotation, team. */
        public static final int SHOT = 4;
        /** tile, aim, recoil x, recoil y. */
        public static final int TURRET = 4;

        public static final int FLAG_MINING = 1;
        public static final int FLAG_BUILDING = 2;
        public static final int FLAG_SHOOTING = 4;

        public int[] units = EMPTY;
        public int[] gone = EMPTY;
        public int[] placed = EMPTY;
        public int[] removed = EMPTY;
        public int[] hurt = EMPTY;
        public int[] shots = EMPTY;
        public int[] turrets = EMPTY;
        /** tile, blend, scale, count, then count times (item, x, y). Variable width. */
        public int[] belts = EMPTY;

        public int agent = -1;

        public int unitCount() {
            return units.length / UNIT;
        }

        public int unit(int index, int field) {
            return units[index * UNIT + field];
        }
    }

    private static final int[] EMPTY = new int[0];
    private static final String[] NO_STRINGS = new String[0];

    public String task = "";
    public String description = "";
    public String map = "";
    public String note = "";
    public String sector;
    public int width, height;
    public int coreX = -1, coreY = -1;
    public final Seq<Frame> frames = new Seq<>();

    /**
     * The world the episode was played on, carried by the replay itself.
     *
     * <p>A replay used to name a map and expect the game to have it. That works for the
     * handful of hand-made maps and fails for every generated sector, which is what the
     * agent now trains on: those have no name, so the mod had nothing to look up and said
     * "Map not found:" with nothing after the colon.
     *
     * <p>The recorder already wrote the whole world into the header, so nothing had to be
     * added to make a replay self-contained. Floors, ores and blocks are palette indices,
     * two bytes each, row-major; rotations are one byte. All four are deflated and
     * base64-encoded.
     */
    public int[] floor, overlay, block, rotation;

    /** Palette index to block name, as the exporter wrote it. */
    public final IntMap<String> palette = new IntMap<>();

    public boolean hasWorld() {
        return floor != null && width > 0 && height > 0;
    }

    public static ReplayFile read(Fi file) throws Exception {
        ReplayFile replay = new ReplayFile();

        try (BufferedReader reader = new BufferedReader(new InputStreamReader(
            new GZIPInputStream(file.read()), StandardCharsets.UTF_8))) {

            String line;
            boolean first = true;
            while ((line = reader.readLine()) != null) {
                if (line.isBlank()) {
                    continue;
                }
                Jval record = Jval.read(line);

                if (first) {
                    replay.readHeader(record);
                    first = false;
                    continue;
                }
                if ("frame".equals(string(record, "type", ""))) {
                    replay.frames.add(readFrame(record));
                }
            }
        }

        if (replay.frames.isEmpty()) {
            throw new IllegalStateException("replay contains no frames");
        }
        return replay;
    }

    private void readHeader(Jval header) {
        task = string(header, "task", "unknown");
        description = string(header, "description", "");
        map = string(header, "map", "");
        note = string(header, "note", "");
        sector = header.get("sector") == null ? null : header.get("sector").asString();
        width = integer(header, "width", 0);
        height = integer(header, "height", 0);

        Jval paletteEntries = header.get("palette");
        if (paletteEntries != null && paletteEntries.isObject()) {
            for (var entry : paletteEntries.asObject()) {
                Jval value = entry.value;
                String name = value.isObject() ? string(value, "name", "") : value.asString();
                if (!name.isEmpty()) {
                    palette.put(Integer.parseInt(entry.key), name);
                }
            }
        }

        int tiles = width * height;
        floor = shorts(header, "floor", tiles);
        overlay = shorts(header, "overlay", tiles);
        block = shorts(header, "block", tiles);
        rotation = bytes(header, "rotation", tiles);

        Jval core = header.get("core");
        if (core != null && core.isArray() && core.asArray().size >= 2) {
            coreX = core.asArray().get(0).asInt();
            coreY = core.asArray().get(1).asInt();
        }
    }

    /** One deflated, base64 plane of little-endian unsigned shorts. */
    private static int[] shorts(Jval header, String key, int count) {
        byte[] raw = inflate(header, key);
        if (raw == null || raw.length < count * 2) {
            return null;
        }
        int[] values = new int[count];
        for (int i = 0; i < count; i++) {
            values[i] = (raw[i * 2] & 0xff) | ((raw[i * 2 + 1] & 0xff) << 8);
        }
        return values;
    }

    private static int[] bytes(Jval header, String key, int count) {
        byte[] raw = inflate(header, key);
        if (raw == null || raw.length < count) {
            return null;
        }
        int[] values = new int[count];
        for (int i = 0; i < count; i++) {
            values[i] = raw[i] & 0xff;
        }
        return values;
    }

    private static byte[] inflate(Jval header, String key) {
        Jval field = header.get(key);
        if (field == null) {
            return null;
        }
        try {
            byte[] packed = Base64.getDecoder().decode(field.asString());
            Inflater inflater = new Inflater();
            inflater.setInput(packed);
            ByteArrayOutputStream out = new ByteArrayOutputStream(packed.length * 4);
            byte[] chunk = new byte[16384];
            while (!inflater.finished()) {
                int read = inflater.inflate(chunk);
                if (read == 0 && (inflater.needsInput() || inflater.needsDictionary())) {
                    break;
                }
                out.write(chunk, 0, read);
            }
            inflater.end();
            return out.toByteArray();
        } catch (Exception ignored) {
            // A replay written by an older recorder has no world in it. Falling back to
            // the map name is better than refusing to open it at all.
            return null;
        }
    }

    private static Frame readFrame(Jval record) {
        Frame frame = new Frame();
        frame.step = integer(record, "step", 0);
        frame.tick = record.get("tick") == null ? 0f : record.get("tick").asFloat();
        frame.wave = integer(record, "wave", 0);
        frame.reward = record.get("reward") == null ? 0f : record.get("reward").asFloat();
        frame.refused = record.get("refused") != null;
        frame.items = record.get("items");

        Jval action = record.get("act");
        if (action != null) {
            frame.actionType = string(action, "t", null);
            frame.block = string(action, "b", null);
            frame.x = integer(action, "x", 0);
            frame.y = integer(action, "y", 0);
            frame.rotation = integer(action, "r", 0);
            readCells(action.get("cells"), frame);
        }

        frame.scene = readScene(record.get("scene"));
        return frame;
    }

    /** A stamp's blocks, whose block name is a string and whose rest is numbers. */
    private static void readCells(Jval cells, Frame frame) {
        if (cells == null || !cells.isArray()) {
            return;
        }
        int count = cells.asArray().size;
        frame.cells = new int[count * 3];
        frame.cellBlocks = new String[count];
        for (int i = 0; i < count; i++) {
            Jval cell = cells.asArray().get(i);
            if (!cell.isArray() || cell.asArray().size < 4) {
                continue;
            }
            frame.cells[i * 3] = cell.asArray().get(0).asInt();
            frame.cells[i * 3 + 1] = cell.asArray().get(1).asInt();
            frame.cells[i * 3 + 2] = cell.asArray().get(3).asInt();
            frame.cellBlocks[i] = cell.asArray().get(2).asString();
        }
    }

    private static Scene readScene(Jval scene) {
        if (scene == null || !scene.isObject()) {
            return null;
        }
        Scene out = new Scene();
        out.agent = integer(scene, "agent", -1);
        out.units = numbers(scene.get("units"));
        out.gone = numbers(scene.get("gone"));
        out.placed = numbers(scene.get("placed"));
        out.removed = numbers(scene.get("removed"));
        out.hurt = numbers(scene.get("hurt"));
        out.shots = numbers(scene.get("shots"));
        out.turrets = numbers(scene.get("turrets"));
        out.belts = numbers(scene.get("belts"));
        return out;
    }

    /** A flat JSON array of numbers, rounded to int. Positions are already in tiles. */
    private static int[] numbers(Jval array) {
        if (array == null || !array.isArray()) {
            return EMPTY;
        }
        int size = array.asArray().size;
        int[] values = new int[size];
        for (int i = 0; i < size; i++) {
            values[i] = Math.round(array.asArray().get(i).asFloat());
        }
        return values;
    }

    private static String string(Jval object, String key, String fallback) {
        Jval value = object.get(key);
        return value == null ? fallback : value.asString();
    }

    private static int integer(Jval object, String key, int fallback) {
        Jval value = object.get(key);
        return value == null ? fallback : value.asInt();
    }
}
