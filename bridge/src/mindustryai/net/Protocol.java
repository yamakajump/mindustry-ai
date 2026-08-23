package mindustryai.net;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.nio.charset.StandardCharsets;

/**
 * Wire format for the bridge socket.
 *
 * <p>Each frame is one type byte, then a four byte big-endian length, then that many bytes
 * of payload. Length prefixing is required because TCP is a stream: without it a reader
 * cannot tell where one message ends, and newline delimiting would break the moment binary
 * payloads arrive.
 *
 * <p>The type byte exists so observation tensors can be sent as raw bytes later without a
 * second protocol. Today only {@link #TYPE_JSON} is used, and reserving one byte now is
 * cheaper than versioning the format afterwards.
 */
public final class Protocol {
    /** UTF-8 encoded JSON. */
    public static final byte TYPE_JSON = 0;

    /** Raw binary, reserved for observation tensors. */
    public static final byte TYPE_BINARY = 1;

    /** Protocol revision, exchanged during the handshake. */
    public static final int VERSION = 1;

    /** Refuse absurd lengths rather than trying to allocate them. */
    private static final int MAX_PAYLOAD = 64 * 1024 * 1024;

    private Protocol() {
    }

    /** One decoded frame. */
    public record Frame(byte type, byte[] payload) {
        public String text() {
            return new String(payload, StandardCharsets.UTF_8);
        }
    }

    public static void writeJson(DataOutputStream out, String json) throws IOException {
        write(out, TYPE_JSON, json.getBytes(StandardCharsets.UTF_8));
    }

    public static void write(DataOutputStream out, byte type, byte[] payload) throws IOException {
        out.writeByte(type);
        out.writeInt(payload.length);
        out.write(payload);
        out.flush();
    }

    /**
     * Read one frame, blocking until it is complete.
     *
     * @throws EOFException if the peer closed the connection cleanly
     */
    public static Frame read(DataInputStream in) throws IOException {
        byte type = in.readByte();
        int length = in.readInt();

        if (length < 0 || length > MAX_PAYLOAD) {
            throw new IOException("refusing frame of length " + length);
        }

        byte[] payload = new byte[length];
        in.readFully(payload);
        return new Frame(type, payload);
    }
}
