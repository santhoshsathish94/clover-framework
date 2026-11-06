package ai.clover.studio;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.ClipData;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.Insets;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.text.TextUtils;
import android.util.Base64;
import android.util.Log;
import android.view.ViewGroup;
import android.view.WindowInsets;
import android.view.WindowInsetsController;
import android.view.WindowManager;
import android.webkit.ConsoleMessage;
import android.webkit.CookieManager;
import android.webkit.JavascriptInterface;
import android.webkit.PermissionRequest;
import android.webkit.ValueCallback;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.FrameLayout;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * The Studio's window on the phone: the same page as on the laptop (clover/web/app.html, or setup.html on the first
 * start), served by the app's own server on 127.0.0.1 and opened with that launch's token as a cookie.
 */
public class MainActivity extends Activity {
    private static final int PICK_FILES = 1;
    private static final int ASK_MIC = 2;
    private static final int ASK_NOTES = 3;
    private static volatile MainActivity current;
    private static volatile boolean awake;

    private FrameLayout root;
    private WebView web;
    private ValueCallback<Uri[]> picked;
    private PermissionRequest micRequest;
    private boolean opened;
    private String origin;
    private boolean clearHistory;

    private final Object audioLock = new Object();
    private volatile boolean recordingAudio = false;
    private AudioRecord audioRecord = null;
    private Thread audioThread = null;
    private ByteArrayOutputStream audioBuffer = null;

    private static final String SPLASH = "<!doctype html><meta name=viewport content='width=device-width,initial-scale=1'>"
            + "<style>html,body{margin:0;height:100%;background:#F7F3E9;color:#39473D;font:16px system-ui,Roboto,sans-serif}"
            + "main{height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:18px}"
            + "svg.c{width:64px;height:64px;animation:s 2.4s linear infinite;transform-origin:50% 50%}"
            + "@keyframes s{to{transform:rotate(360deg)}}</style>"
            + "<main><svg class=c viewBox='0 0 24 24'><defs><linearGradient id=lg x1=0 y1=0 x2=1 y2=1>"
            + "<stop offset='0%' stop-color='#FFF078'/><stop offset='55%' stop-color='#F8CF1E'/><stop offset='100%' stop-color='#D89606'/></linearGradient>"
            + "<path id=lf fill='url(#lg)' d='M12 11.4 C9.6 8.8 9.2 5.4 10.9 4.3 C11.6 3.8 12 4.5 12 5.1 C12 4.5 12.4 3.8 13.1 4.3 C14.8 5.4 14.4 8.8 12 11.4 Z'/></defs>"
            + "<use href='#lf'/><use href='#lf' transform='rotate(72 12 12)' opacity='.92'/>"
            + "<use href='#lf' transform='rotate(144 12 12)' opacity='.84'/><use href='#lf' transform='rotate(216 12 12)' opacity='.76'/>"
            + "<use href='#lf' transform='rotate(288 12 12)' opacity='.68'/><circle cx=12 cy=12 r='1.3' fill='#39473D'/></svg>"
            + "<div>{text}</div></main>";

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        current = this;
        root = new FrameLayout(this);
        root.setBackgroundColor(Color.parseColor("#F7F3E9"));
        web = new WebView(this);
        root.addView(web, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT));
        root.setOnApplyWindowInsetsListener((v, windowInsets) -> {
            if (Build.VERSION.SDK_INT >= 30) {
                Insets bars = windowInsets.getInsets(
                        WindowInsets.Type.systemBars()
                                | WindowInsets.Type.displayCutout()
                                | WindowInsets.Type.ime());
                v.setPadding(bars.left, bars.top, bars.right, bars.bottom);
                return WindowInsets.CONSUMED;
            } else {
                v.setPadding(
                        windowInsets.getSystemWindowInsetLeft(),
                        windowInsets.getSystemWindowInsetTop(),
                        windowInsets.getSystemWindowInsetRight(),
                        windowInsets.getSystemWindowInsetBottom());
                return windowInsets.consumeSystemWindowInsets();
            }
        });
        setContentView(root);
        if (Build.VERSION.SDK_INT >= 30) {
            WindowInsetsController ic = getWindow().getInsetsController();
            if (ic != null) {
                int mask = WindowInsetsController.APPEARANCE_LIGHT_STATUS_BARS
                        | WindowInsetsController.APPEARANCE_LIGHT_NAVIGATION_BARS;
                ic.setSystemBarsAppearance(mask, mask);
            }
        }
        root.requestApplyInsets();
        applyAwake();
        WebSettings s = web.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);
        s.setMediaPlaybackRequiresUserGesture(false);  // Clover's spoken answers start by themselves, as on the laptop
        s.setAllowFileAccess(false);
        s.setAllowContentAccess(false);
        s.setTextZoom(100);
        web.addJavascriptInterface(new CloverAndroidBridge(), "CloverAndroidBridge");
        web.setWebViewClient(new WebViewClient() {
            @Override
            public void onPageFinished(WebView view, String url) {
                if (clearHistory && origin != null && url != null && url.startsWith(origin)) {
                    clearHistory = false;
                    view.clearHistory();  // the start screen is not a page to go back to
                }
            }

            @Override
            public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest request) {
                Uri u = request.getUrl();
                if ("127.0.0.1".equals(u.getHost())) {
                    return false;
                }
                try {  // links in answers open in the phone's browser
                    startActivity(new Intent(Intent.ACTION_VIEW, u));
                } catch (Exception ignored) {
                    // no browser
                }
                return true;
            }
        });
        web.setWebChromeClient(new WebChromeClient() {
            @Override
            public boolean onShowFileChooser(WebView view, ValueCallback<Uri[]> callback, FileChooserParams params) {
                if (picked != null) {
                    picked.onReceiveValue(null);
                }
                picked = callback;
                Intent i = new Intent(Intent.ACTION_GET_CONTENT).addCategory(Intent.CATEGORY_OPENABLE).setType("*/*");
                i.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, params.getMode() == FileChooserParams.MODE_OPEN_MULTIPLE);
                try {
                    startActivityForResult(Intent.createChooser(i, "Add files"), PICK_FILES);
                } catch (Exception e) {
                    picked = null;
                    return false;
                }
                return true;
            }

            @Override
            public void onPermissionRequest(PermissionRequest request) {
                // the microphone for voice messages; only the app's own page may ask
                if (!"127.0.0.1".equals(request.getOrigin().getHost())) {
                    request.deny();
                    return;
                }
                if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
                    request.grant(request.getResources());
                } else {
                    micRequest = request;
                    requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO}, ASK_MIC);
                }
            }

            @Override
            public boolean onConsoleMessage(ConsoleMessage m) {
                Log.i(StudioService.TAG, "page: " + m.message() + " (" + m.sourceId() + ":" + m.lineNumber() + ")");
                return true;
            }
        });
        if (Build.VERSION.SDK_INT >= 33
                && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.POST_NOTIFICATIONS}, ASK_NOTES);
        }
        startForegroundService(new Intent(this, StudioService.class));
        page(SPLASH.replace("{text}", "Starting Clover"));
        waitForServer();
    }

    private void page(String html) {
        web.loadDataWithBaseURL(null, html, "text/html", "utf-8", null);
    }

    private void waitForServer() {
        new Thread(() -> {
            long t0 = System.currentTimeMillis();
            while (!isFinishing()) {
                if (StudioService.error != null) {
                    fail("Clover could not start: " + StudioService.error);
                    return;
                }
                int p = StudioService.port;
                String t = StudioService.token;
                if (p > 0 && t != null && ping(p, t)) {
                    runOnUiThread(() -> open(p, t));
                    return;
                }
                if (System.currentTimeMillis() - t0 > 180_000) {
                    fail("Clover's server did not start. Close Clover from its notification and open it again.");
                    return;
                }
                try {
                    Thread.sleep(250);
                } catch (InterruptedException e) {
                    return;
                }
            }
        }, "clover-wait").start();
    }

    private static boolean ping(int port, String token) {
        try {
            HttpURLConnection c = (HttpURLConnection) new URL("http://127.0.0.1:" + port + "/api/setup").openConnection();
            c.setConnectTimeout(1000);
            c.setReadTimeout(3000);
            c.setRequestProperty("X-Clover-Token", token);
            int code = c.getResponseCode();
            c.disconnect();
            return code == 200;
        } catch (IOException e) {
            return false;
        }
    }

    private void fail(String why) {
        runOnUiThread(() -> page(SPLASH.replace("animation:s 2.4s linear infinite", "")
                .replace("{text}", TextUtils.htmlEncode(why))));
    }

    private void open(int port, String token) {
        if (opened) {
            return;
        }
        opened = true;
        origin = "http://127.0.0.1:" + port;
        CookieManager cm = CookieManager.getInstance();
        cm.setAcceptCookie(true);
        cm.setCookie(origin, "clover_token=" + token + "; Path=/; HttpOnly; SameSite=Strict", ok -> {
            cm.flush();
            clearHistory = true;
            web.loadUrl(origin + "/");
        });
    }

    @Override
    public void onBackPressed() {
        // the page first closes what it has open (a settings page, the chat list); else Clover goes to the background
        // and keeps running (Stop in its notification ends it)
        String url = web != null ? web.getUrl() : null;
        if (origin == null || url == null || !url.startsWith(origin)) {
            moveTaskToBack(true);
            return;
        }
        web.evaluateJavascript("window.cloverBack ? window.cloverBack() : false", handled -> {
            if (!"true".equals(handled)) {
                moveTaskToBack(true);
            }
        });
    }

    @Override
    protected void onActivityResult(int request, int result, Intent data) {
        super.onActivityResult(request, result, data);
        if (request != PICK_FILES || picked == null) {
            return;
        }
        Uri[] uris = null;
        if (result == RESULT_OK && data != null) {
            ClipData clip = data.getClipData();
            if (clip != null) {
                uris = new Uri[clip.getItemCount()];
                for (int i = 0; i < clip.getItemCount(); i++) {
                    uris[i] = clip.getItemAt(i).getUri();
                }
            } else if (data.getData() != null) {
                uris = new Uri[]{data.getData()};
            }
        }
        picked.onReceiveValue(uris);
        picked = null;
    }

    @Override
    public void onRequestPermissionsResult(int request, String[] permissions, int[] results) {
        if (request == ASK_MIC && micRequest != null) {
            if (results.length > 0 && results[0] == PackageManager.PERMISSION_GRANTED) {
                micRequest.grant(micRequest.getResources());
            } else {
                micRequest.deny();
            }
            micRequest = null;
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (root != null) {
            root.requestApplyInsets();
        }
    }

    @Override
    protected void onDestroy() {
        if (current == this) {
            current = null;
        }
        super.onDestroy();
    }

    static void finishAll() {
        MainActivity a = current;
        if (a != null) {
            a.runOnUiThread(a::finishAndRemoveTask);
        }
    }

    /** On while the first download or the model's load runs: with the screen off, Android ranked Clover below the apps
     *  it closed for memory during the S23's first load and closed Clover too (2026-11-04). */
    static void keepScreenOn(boolean on) {
        awake = on;
        MainActivity a = current;
        if (a != null) {
            a.runOnUiThread(a::applyAwake);
        }
    }

    private void applyAwake() {
        if (awake) {
            getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        } else {
            getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        }
    }

    private static byte[] pcm16ToWav(byte[] pcm, int sampleRate) {
        ByteBuffer hdr = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN);
        hdr.put(new byte[]{'R', 'I', 'F', 'F'});
        hdr.putInt(36 + pcm.length);
        hdr.put(new byte[]{'W', 'A', 'V', 'E', 'f', 'm', 't', ' '});
        hdr.putInt(16);
        hdr.putShort((short) 1);
        hdr.putShort((short) 1);
        hdr.putInt(sampleRate);
        hdr.putInt(sampleRate * 2);
        hdr.putShort((short) 2);
        hdr.putShort((short) 16);
        hdr.put(new byte[]{'d', 'a', 't', 'a'});
        hdr.putInt(pcm.length);
        byte[] out = new byte[44 + pcm.length];
        System.arraycopy(hdr.array(), 0, out, 0, 44);
        System.arraycopy(pcm, 0, out, 44, pcm.length);
        return out;
    }

    public final class CloverAndroidBridge {
        @SuppressLint("MissingPermission")
        @JavascriptInterface
        public String startAudioCapture() {
            if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
                runOnUiThread(() -> requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO}, ASK_MIC));
                return "PERMISSION_REQUESTED";
            }
            synchronized (audioLock) {
                if (recordingAudio) {
                    return "OK";
                }
                int rate = 16000;
                int minBuf = AudioRecord.getMinBufferSize(
                        rate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
                if (minBuf <= 0) {
                    minBuf = rate * 2;
                }
                try {
                    audioRecord = new AudioRecord(
                            MediaRecorder.AudioSource.MIC,
                            rate,
                            AudioFormat.CHANNEL_IN_MONO,
                            AudioFormat.ENCODING_PCM_16BIT,
                            Math.max(minBuf * 2, rate * 2));
                    if (audioRecord.getState() != AudioRecord.STATE_INITIALIZED) {
                        audioRecord.release();
                        audioRecord = null;
                        return "ERROR:UNINITIALIZED";
                    }
                    audioBuffer = new ByteArrayOutputStream(rate * 10);
                    recordingAudio = true;
                    audioRecord.startRecording();
                } catch (Throwable e) {
                    recordingAudio = false;
                    if (audioRecord != null) {
                        try {
                            audioRecord.release();
                        } catch (Throwable ignored) {
                        }
                        audioRecord = null;
                    }
                    return "ERROR:" + e.getMessage();
                }
                final int bufSize = Math.max(minBuf, 2048);
                audioThread = new Thread(() -> {
                    byte[] buf = new byte[bufSize];
                    while (recordingAudio) {
                        AudioRecord ar = audioRecord;
                        if (ar == null) {
                            break;
                        }
                        int n = ar.read(buf, 0, buf.length);
                        if (n > 0) {
                            synchronized (audioLock) {
                                if (audioBuffer != null) {
                                    audioBuffer.write(buf, 0, n);
                                }
                            }
                        }
                    }
                }, "clover-mic-pcm");
                audioThread.start();
                return "OK";
            }
        }

        @JavascriptInterface
        public String stopAudioCapture() {
            Thread t;
            AudioRecord ar;
            synchronized (audioLock) {
                if (!recordingAudio) {
                    return "";
                }
                recordingAudio = false;
                t = audioThread;
                audioThread = null;
                ar = audioRecord;
                audioRecord = null;
            }
            if (ar != null) {
                try {
                    ar.stop();
                } catch (Throwable ignored) {
                }
                try {
                    ar.release();
                } catch (Throwable ignored) {
                }
            }
            if (t != null) {
                try {
                    t.join(600);
                } catch (InterruptedException ignored) {
                }
            }
            byte[] pcm;
            synchronized (audioLock) {
                pcm = audioBuffer != null ? audioBuffer.toByteArray() : new byte[0];
                audioBuffer = null;
            }
            if (pcm.length == 0) {
                return "";
            }
            return Base64.encodeToString(pcm16ToWav(pcm, 16000), Base64.NO_WRAP);
        }

        @JavascriptInterface
        public void startGoogleSignIn(String authUrl, String nonce) {
            if (authUrl == null || authUrl.isEmpty()) {
                return;
            }
            runOnUiThread(() -> {
                try {
                    Intent i = new Intent(Intent.ACTION_VIEW, Uri.parse(authUrl));
                    startActivity(i);
                } catch (Exception ignored) {
                }
            });
        }
    }
}
