package mindustryai.net;

import arc.util.Log;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.TimeUnit;

/**
 * Accepts one agent connection and shuttles messages between it and the game thread.
 *
 * <p>Threading is the whole point of this class. Mindustry state may only be touched from
 * the game thread, and socket reads block, so the two cannot be the same thread. Requests
 * arrive on the network thread and are handed to the game thread through a queue; replies
 * travel back the same way.
 *
 * <p>One connection at a time, deliberately. An environment is one agent talking to one
 * game, and allowing several would raise questions about who gets to step the world that
 * have no good answer.
 *
 * <p>Bound to loopback only. This carries no authentication and executes whatever it is
 * told, so it has no business listening on a public interface.
 */
public class BridgeServer {
    private final int port;
    private final BlockingQueue<String> requests = new ArrayBlockingQueue<>(16);
    private final BlockingQueue<Reply> replies = new ArrayBlockingQueue<>(16);

    /**
     * One answer: a JSON frame, optionally followed by a binary frame.
     *
     * <p>Two frames rather than one mixed payload, so the client can read the shape and
     * dtype out of the JSON before deciding how to interpret the bytes that follow.
     */
    public record Reply(String json, byte[] binary) {
    }

    private volatile ServerSocket serverSocket;
    private volatile Socket client;
    private volatile boolean running;
    private Thread acceptThread;

    /**
     * Incremented for every accepted connection.
     *
     * <p>Work started for one agent must never have its reply delivered to the next one.
     * A step spans many ticks, so an agent can disconnect while its step is still running;
     * without this counter the late reply would sit in the queue and be handed to whoever
     * connects next, silently shifting every subsequent request and reply by one.
     */
    private final java.util.concurrent.atomic.AtomicInteger session =
        new java.util.concurrent.atomic.AtomicInteger();

    public BridgeServer(int port) {
        this.port = port;
    }

    /** Begin accepting connections. Returns immediately. */
    public void start() throws IOException {
        serverSocket = new ServerSocket(port, 1, InetAddress.getLoopbackAddress());
        running = true;

        acceptThread = new Thread(this::acceptLoop, "mindustry-ai-bridge");
        acceptThread.setDaemon(true);
        acceptThread.start();

        Log.info("[mindustry-ai] listening on 127.0.0.1:@", port);
    }

    private void acceptLoop() {
        while (running) {
            try (Socket socket = serverSocket.accept()) {
                socket.setTcpNoDelay(true);
                client = socket;
                session.incrementAndGet();
                Log.info("[mindustry-ai] agent connected, session @", session.get());
                serve(socket);
            } catch (IOException e) {
                if (running) {
                    Log.info("[mindustry-ai] agent disconnected: @", e.getMessage());
                }
            } finally {
                client = null;
                // Drop anything left over so a reconnecting agent starts clean rather
                // than receiving a reply meant for its predecessor.
                requests.clear();
                replies.clear();
            }
        }
    }

    private void serve(Socket socket) throws IOException {
        DataInputStream in = new DataInputStream(socket.getInputStream());
        DataOutputStream out = new DataOutputStream(socket.getOutputStream());

        while (running && !socket.isClosed()) {
            Protocol.Frame frame = Protocol.read(in);
            if (frame.type() != Protocol.TYPE_JSON) {
                throw new IOException("unexpected frame type " + frame.type());
            }

            // Blocks until the game thread has produced a reply. A step spanning many
            // ticks legitimately takes a while, so there is no timeout here: a hung game
            // shows up as a hung agent, which is the honest signal.
            try {
                requests.put(frame.text());
                Reply reply = replies.take();
                Protocol.writeJson(out, reply.json());
                if (reply.binary() != null) {
                    Protocol.write(out, Protocol.TYPE_BINARY, reply.binary());
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return;
            }
        }
    }

    /**
     * Take the next pending request, or null if none is waiting.
     * Called from the game thread only.
     */
    public String pollRequest() {
        return requests.poll();
    }

    /** Hand a reply back to the network thread. Called from the game thread only. */
    public void reply(String json) {
        reply(json, null);
    }

    /**
     * Hand back a reply with a binary payload. Called from the game thread only.
     *
     * <p>The array is copied. The encoder reuses its buffer between calls, and handing
     * the live one to another thread would work right up until it did not.
     */
    public void reply(String json, byte[] binary) {
        Reply payload = new Reply(json, binary == null ? null : binary.clone());
        try {
            if (!replies.offer(payload, 5, TimeUnit.SECONDS)) {
                Log.err("[mindustry-ai] reply queue full, dropping response");
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    public boolean hasClient() {
        return client != null;
    }

    /** Identifier of the current connection. See the field comment on {@code session}. */
    public int session() {
        return session.get();
    }

    public int port() {
        return port;
    }

    public void stop() {
        running = false;
        try {
            if (client != null) {
                client.close();
            }
            if (serverSocket != null) {
                serverSocket.close();
            }
        } catch (IOException ignored) {
            // Shutting down anyway.
        }
    }
}
