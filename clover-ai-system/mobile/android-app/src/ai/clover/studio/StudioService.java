package ai.clover.studio;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.net.wifi.WifiManager;
import android.os.Build;
import android.os.IBinder;
import android.os.PowerManager;
import android.util.Log;

import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import java.io.IOException;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.security.SecureRandom;

/**
 * Keeps Clover running: the laptop's Studio server (Python, clover/server.py) and its llama.cpp, in this app's process,
 * while a notification shows that it runs (and the first download's progress). Stop in the notification ends both.
 */
public class StudioService extends Service {
    static final String TAG = "Clover";
    private static final String CHANNEL = "clover";
    private static final int NOTE = 1;
    // a fixed port keeps the page's origin, and so its saved state, the same at every start; the next ones if taken
    private static final int FIRST_PORT = 47651;

    static volatile int port = 0;
    static volatile String token = null;
    static volatile String error = null;
    private static StudioService self;

    private PowerManager.WakeLock wake;
    private WifiManager.WifiLock wifi;

    @Override
    public void onCreate() {
        super.onCreate();
        self = this;
        NotificationChannel ch = new NotificationChannel(CHANNEL, "Clover AI Studio", NotificationManager.IMPORTANCE_LOW);
        ch.setShowBadge(false);
        getSystemService(NotificationManager.class).createNotificationChannel(ch);
        Notification n = note("Clover runs on this phone", -1);
        if (Build.VERSION.SDK_INT >= 34) {
            startForeground(NOTE, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE);
        } else {
            startForeground(NOTE, n);
        }
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent != null && "stop".equals(intent.getAction())) {
            stopAll();
            return START_NOT_STICKY;
        }
        start();
        return START_NOT_STICKY;
    }

    private synchronized void start() {
        if (token != null) {
            return;  // running in this process already
        }
        byte[] b = new byte[24];
        new SecureRandom().nextBytes(b);
        StringBuilder t = new StringBuilder();
        for (byte x : b) {
            t.append(String.format("%02x", x));
        }
        token = t.toString();
        port = freePort();
        final String soc = Build.VERSION.SDK_INT >= 31 ? (Build.SOC_MANUFACTURER + " " + Build.SOC_MODEL) : Build.HARDWARE;
        new Thread(() -> {
            try {
                if (!Python.isStarted()) {
                    Python.start(new AndroidPlatform(this));
                }
                Python.getInstance().getModule("android_main").callAttr("serve", getFilesDir().getAbsolutePath(),
                        getApplicationInfo().nativeLibraryDir, port, token, Build.MANUFACTURER + " " + Build.MODEL, soc);
            } catch (Throwable e) {
                error = String.valueOf(e);
                Log.e(TAG, "Clover's server did not start", e);
            }
        }, "clover-start").start();
    }

    private static int freePort() {
        for (int p = FIRST_PORT; p < FIRST_PORT + 20; p++) {
            try (ServerSocket s = new ServerSocket(p, 1, InetAddress.getByName("127.0.0.1"))) {
                return p;
            } catch (IOException ignored) {
                // taken: the next one
            }
        }
        return FIRST_PORT;
    }

    /** From Python (server.Setup): the download's progress; percent -1 for a plain message. */
    public static void progress(String text, int percent) {
        StudioService s = self;
        if (s == null) {
            return;
        }
        s.getSystemService(NotificationManager.class).notify(NOTE, s.note(text, percent));
        s.hold(percent >= 0 && percent < 100);
        MainActivity.keepScreenOn(percent >= 0 && percent < 100);
    }

    /** A download or the model's load keeps the CPU and Wi-Fi awake with the screen off; nothing is held otherwise. */
    private synchronized void hold(boolean on) {
        if (on && wake == null) {
            wake = getSystemService(PowerManager.class).newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "clover:download");
            wake.acquire(3 * 60 * 60 * 1000L);
            wifi = getSystemService(WifiManager.class).createWifiLock(WifiManager.WIFI_MODE_FULL_HIGH_PERF, "clover:download");
            wifi.acquire();
        } else if (!on && wake != null) {
            if (wake.isHeld()) {
                wake.release();
            }
            if (wifi != null && wifi.isHeld()) {
                wifi.release();
            }
            wake = null;
            wifi = null;
        }
    }

    private Notification note(String text, int percent) {
        Intent open = new Intent(this, MainActivity.class).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP);
        PendingIntent pOpen = PendingIntent.getActivity(this, 0, open, PendingIntent.FLAG_IMMUTABLE);
        PendingIntent pStop = PendingIntent.getService(this, 1, new Intent(this, StudioService.class).setAction("stop"),
                PendingIntent.FLAG_IMMUTABLE);
        Notification.Builder b = new Notification.Builder(this, CHANNEL)
                .setSmallIcon(R.drawable.ic_stat)
                .setContentTitle("Clover AI Studio")
                .setContentText(text)
                .setContentIntent(pOpen)
                .setOngoing(true)
                .setOnlyAlertOnce(true)
                .addAction(new Notification.Action.Builder(null, "Stop", pStop).build());
        if (percent >= 0 && percent < 100) {
            b.setProgress(100, percent, false);
        }
        return b.build();
    }

    /** Ends llama.cpp first (it is a separate process holding the model's memory), then this app's process. */
    private void stopAll() {
        try {
            if (Python.isStarted()) {
                Python.getInstance().getModule("android_main").callAttr("stop");
            }
        } catch (Throwable e) {
            Log.w(TAG, "stop", e);
        }
        hold(false);
        stopForeground(STOP_FOREGROUND_REMOVE);
        stopSelf();
        MainActivity.finishAll();
        android.os.Process.killProcess(android.os.Process.myPid());
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
}
