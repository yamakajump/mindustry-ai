package mindustryai.replay;

import arc.files.Fi;
import arc.struct.Seq;
import arc.util.serialization.Jval;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.zip.GZIPInputStream;

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
    }

    public String task = "";
    public String description = "";
    public String map = "";
    public String note = "";
    public String sector;
    public int width, height;
    public int coreX = -1, coreY = -1;
    public final Seq<Frame> frames = new Seq<>();

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

        Jval core = header.get("core");
        if (core != null && core.isArray() && core.asArray().size >= 2) {
            coreX = core.asArray().get(0).asInt();
            coreY = core.asArray().get(1).asInt();
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
        }
        return frame;
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
