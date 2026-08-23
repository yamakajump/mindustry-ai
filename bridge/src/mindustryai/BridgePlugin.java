package mindustryai;

import arc.Core;
import arc.util.CommandHandler;
import arc.util.Log;
import mindustry.Vars;
import mindustry.mod.Plugin;
import mindustryai.net.BridgeServer;
import mindustryai.net.StepLoop;

import java.io.IOException;

/**
 * Entry point for the mindustry-ai bridge.
 *
 * <p>This class registers commands and wires components together. It deliberately holds no
 * game logic and no strategy: anything resembling a judgment about what a good move is
 * belongs on the Python side, where it can be learned rather than hardcoded. See
 * {@code docs/decisions/0002-three-process-architecture.md}.
 */
public class BridgePlugin extends Plugin {
    public static final String VERSION = "0.1.0";

    /** Overridden with -Dmindustryai.port=N so parallel instances do not collide. */
    private static final String PORT_PROPERTY = "mindustryai.port";
    private static final int DEFAULT_PORT = 7654;

    private final Clock clock = new Clock();
    private final Benchmark benchmark = new Benchmark();
    private BridgeServer server;
    private StepLoop stepLoop;

    @Override
    public void init() {
        boolean clockReady = clock.install();
        Log.info("[mindustry-ai] bridge @ loaded, clock=@", VERSION, clockReady ? "ok" : "degraded");

        int port = Integer.getInteger(PORT_PROPERTY, DEFAULT_PORT);
        server = new BridgeServer(port);
        stepLoop = new StepLoop(server, clock);
        stepLoop.install();

        try {
            server.start();
        } catch (IOException e) {
            Log.err("[mindustry-ai] could not listen on port @: @", port, e.getMessage());
        }
    }

    @Override
    public void registerServerCommands(CommandHandler handler) {
        handler.register("bridge-status", "Report bridge state.", args ->
            // state.tick is game time in ticks. Comparing its delta against wall-clock
            // time is the only unambiguous proof that acceleration advances the world
            // rather than merely spinning the application loop faster.
            Log.info("bridge ready version=@ clock=@ speed=@ fps=@ tick=@ wave=@ playing=@",
                VERSION,
                clock.isOperational() ? "ok" : "degraded",
                clock.speed(),
                Core.graphics.getFramesPerSecond(),
                String.format(java.util.Locale.ROOT, "%.1f", Vars.state.tick),
                Vars.state.wave,
                Vars.state.isPlaying())
        );

        handler.register("bridge-bench", "<seconds>", "Measure simulation throughput.", args -> {
            try {
                benchmark.start(Integer.parseInt(args[0]));
            } catch (NumberFormatException e) {
                Log.err("bench window must be an integer number of seconds");
            }
        });

        handler.register("bridge-port", "Report the agent socket port.", args ->
            Log.info("bridge port=@ connected=@", server.port(), server.hasClient())
        );

        handler.register("bridge-speed", "<multiplier|max>", "Set simulation speed.", args -> {
            String raw = args[0];
            if (raw.equalsIgnoreCase("max")) {
                clock.setSpeed(Integer.MAX_VALUE);
                return;
            }
            try {
                clock.setSpeed(Integer.parseInt(raw));
            } catch (NumberFormatException e) {
                Log.err("speed must be an integer or 'max'");
            }
        });
    }
}
