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
    private final BlockingQueue<String> replies = new ArrayBlockingQueue<>(16);

    private volatile ServerSocket serverSocket;
    private volatile Socket client;
    private volatile boolean running;
    private Thread acceptThread;

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
                Log.info("[mindustry-ai] agent connected");
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
                String reply = replies.take();
                Protocol.writeJson(out, reply);
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
        try {
            if (!replies.offer(json, 5, TimeUnit.SECONDS)) {
                Log.err("[mindustry-ai] reply queue full, dropping response");
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    public boolean hasClient() {
        return client != null;
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
