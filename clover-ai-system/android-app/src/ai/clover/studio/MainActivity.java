package ai.clover.studio;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.speech.RecognizerIntent;
import android.speech.tts.TextToSpeech;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.webkit.JavascriptInterface;
import android.webkit.PermissionRequest;
import android.webkit.ValueCallback;
import android.webkit.WebChromeClient;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.Toast;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Locale;

public class MainActivity extends Activity {

    private static final String EXCLUSIVE_APP_PACKAGE = "ai.clover.studio";
    private static final String EXCLUSIVE_APP_TOKEN =
            "clover-ai-studio-s23-exclusive-v1-9f4e2a7c8b1d6035e4a9c2f7b8d1e604";
    private static final String LOCAL_S23_BASE_URL = "http://127.0.0.1:8765";

    private static final int REQ_FILE_CHOOSER = 1001;
    private static final int REQ_SPEECH_INPUT = 1002;
    private static final int REQ_PERMISSIONS = 1003;

    private WebView webView;
    private ValueCallback<Uri[]> filePathCallback;
    private TextToSpeech tts;
    private boolean ttsReady = false;

    @SuppressLint({"SetJavaScriptEnabled", "AddJavascriptInterface"})
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        Window window = getWindow();
        if (window != null) {
            window.addFlags(WindowManager.LayoutParams.FLAG_DRAWS_SYSTEM_BAR_BACKGROUNDS);
            window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
            window.setStatusBarColor(Color.parseColor("#FAF8F2"));
            window.setNavigationBarColor(Color.parseColor("#FAF8F2"));
            if (Build.VERSION.SDK_INT >= 23) {
                View decor = window.getDecorView();
                decor.setSystemUiVisibility(View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR);
            }
        }

        tts = new TextToSpeech(this, new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int status) {
                if (status == TextToSpeech.SUCCESS) {
                    tts.setLanguage(Locale.US);
                    tts.setSpeechRate(0.98f);
                    ttsReady = true;
                }
            }
        });

        webView = new WebView(this);
        setContentView(webView);

        WebSettings settings = webView.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);
        settings.setDatabaseEnabled(true);
        settings.setAllowFileAccess(true);
        settings.setAllowContentAccess(true);
        settings.setAllowFileAccessFromFileURLs(true);
        settings.setAllowUniversalAccessFromFileURLs(true);
        settings.setMediaPlaybackRequiresUserGesture(false);
        settings.setMixedContentMode(WebSettings.MIXED_CONTENT_ALWAYS_ALLOW);
        settings.setUserAgentString(
                "CloverAIStudio-Android/1.0 (Samsung-Galaxy-S23; Snapdragon-8-Gen-2; ai.clover.studio)"
        );

        webView.addJavascriptInterface(new CloverAndroidBridge(), "CloverAndroidBridge");

        webView.setWebViewClient(new WebViewClient() {
            @Override
            public boolean shouldOverrideUrlLoading(WebView view, String url) {
                if (url != null && (url.startsWith("http://") || url.startsWith("https://"))) {
                    Intent i = new Intent(Intent.ACTION_VIEW, Uri.parse(url));
                    startActivity(i);
                    return true;
                }
                return false;
            }
        });

        webView.setWebChromeClient(new WebChromeClient() {
            @Override
            public void onPermissionRequest(final PermissionRequest request) {
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        request.grant(request.getResources());
                    }
                });
            }

            @Override
            public boolean onShowFileChooser(
                    WebView webView,
                    ValueCallback<Uri[]> filePathCallbackParam,
                    FileChooserParams fileChooserParams
            ) {
                if (filePathCallback != null) {
                    filePathCallback.onReceiveValue(null);
                }
                filePathCallback = filePathCallbackParam;
                try {
                    Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
                    intent.addCategory(Intent.CATEGORY_OPENABLE);
                    intent.setType("*/*");
                    intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true);
                    startActivityForResult(Intent.createChooser(intent, "Attach files to Clover AI Studio"), REQ_FILE_CHOOSER);
                } catch (Exception e) {
                    filePathCallback = null;
                    return false;
                }
                return true;
            }
        });

        webView.loadUrl("file:///android_asset/index.html");
    }

    private boolean hasPerm(String perm) {
        if (Build.VERSION.SDK_INT < 23) return true;
        return checkSelfPermission(perm) == PackageManager.PERMISSION_GRANTED;
    }

    public class CloverAndroidBridge {

        @JavascriptInterface
        public String getAppPackage() {
            return EXCLUSIVE_APP_PACKAGE;
        }

        @JavascriptInterface
        public String callLocalS23Api(String method, String endpoint, String jsonBody) {
            HttpURLConnection conn = null;
            try {
                String path = endpoint.startsWith("/") ? endpoint : ("/" + endpoint);
                URL url = new URL(LOCAL_S23_BASE_URL + path);
                conn = (HttpURLConnection) url.openConnection();
                conn.setRequestMethod(method != null ? method.toUpperCase(Locale.US) : "GET");
                conn.setConnectTimeout(6000);
                conn.setReadTimeout(120000);
                conn.setRequestProperty("Content-Type", "application/json; charset=utf-8");
                conn.setRequestProperty("X-Clover-App-Package", EXCLUSIVE_APP_PACKAGE);
                conn.setRequestProperty("X-Clover-App-Token", EXCLUSIVE_APP_TOKEN);
                conn.setRequestProperty(
                        "User-Agent",
                        "CloverAIStudio-Android/1.0 (Samsung-Galaxy-S23; Snapdragon-8-Gen-2; ai.clover.studio)"
                );

                if ("POST".equalsIgnoreCase(method) && jsonBody != null) {
                    conn.setDoOutput(true);
                    byte[] outBytes = jsonBody.getBytes(StandardCharsets.UTF_8);
                    conn.setFixedLengthStreamingMode(outBytes.length);
                    OutputStream os = conn.getOutputStream();
                    os.write(outBytes);
                    os.flush();
                    os.close();
                }

                int code = conn.getResponseCode();
                InputStream is = (code >= 200 && code < 400) ? conn.getInputStream() : conn.getErrorStream();
                if (is == null) {
                    return "{\"error\":\"empty_response\",\"http_code\":" + code + "}";
                }
                BufferedReader reader = new BufferedReader(new InputStreamReader(is, StandardCharsets.UTF_8));
                StringBuilder sb = new StringBuilder();
                String line;
                while ((line = reader.readLine()) != null) {
                    sb.append(line).append('\n');
                }
                reader.close();
                return sb.toString().trim();
            } catch (Exception ex) {
                String msg = ex.getMessage() != null ? ex.getMessage().replace("\"", "'") : "connection_failed";
                return "{\"error\":\"s23_local_unreachable\",\"message\":\"" + msg + "\"}";
            } finally {
                if (conn != null) {
                    conn.disconnect();
                }
            }
        }

        @JavascriptInterface
        public String getOsPermissionsStatus() {
            boolean mic = hasPerm(Manifest.permission.RECORD_AUDIO);
            boolean cam = hasPerm(Manifest.permission.CAMERA);
            boolean files = hasPerm(Manifest.permission.READ_EXTERNAL_STORAGE)
                    || hasPerm("android.permission.READ_MEDIA_IMAGES")
                    || hasPerm("android.permission.READ_MEDIA_VIDEO")
                    || hasPerm("android.permission.READ_MEDIA_AUDIO");
            return "{\"microphone\":" + mic + ",\"camera_vision\":" + cam + ",\"files\":" + files + ",\"internet\":true}";
        }

        @JavascriptInterface
        public void requestOsPermission(final String group) {
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    if (Build.VERSION.SDK_INT < 23) return;
                    ArrayList<String> perms = new ArrayList<String>();
                    if ("microphone".equalsIgnoreCase(group) || "all".equalsIgnoreCase(group)) {
                        perms.add(Manifest.permission.RECORD_AUDIO);
                    }
                    if ("camera_vision".equalsIgnoreCase(group) || "all".equalsIgnoreCase(group)) {
                        perms.add(Manifest.permission.CAMERA);
                    }
                    if ("files".equalsIgnoreCase(group) || "all".equalsIgnoreCase(group)) {
                        if (Build.VERSION.SDK_INT >= 33) {
                            perms.add("android.permission.READ_MEDIA_IMAGES");
                            perms.add("android.permission.READ_MEDIA_VIDEO");
                            perms.add("android.permission.READ_MEDIA_AUDIO");
                        } else {
                            perms.add(Manifest.permission.READ_EXTERNAL_STORAGE);
                            perms.add(Manifest.permission.WRITE_EXTERNAL_STORAGE);
                        }
                    }
                    if (!perms.isEmpty()) {
                        requestPermissions(perms.toArray(new String[0]), REQ_PERMISSIONS);
                    }
                }
            });
        }

        @JavascriptInterface
        public void startNativeVoiceRecognition() {
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    if (!hasPerm(Manifest.permission.RECORD_AUDIO)) {
                        if (Build.VERSION.SDK_INT >= 23) {
                            requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO}, REQ_PERMISSIONS);
                        }
                        return;
                    }
                    try {
                        Intent intent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
                        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
                        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault());
                        intent.putExtra(RecognizerIntent.EXTRA_PROMPT, "Speak to Clover AI Studio (Samsung S23)");
                        startActivityForResult(intent, REQ_SPEECH_INPUT);
                    } catch (Exception ex) {
                        Toast.makeText(MainActivity.this, "Voice input unavailable on this device", Toast.LENGTH_SHORT).show();
                    }
                }
            });
        }

        @JavascriptInterface
        public void speakText(final String text) {
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    if (ttsReady && tts != null && text != null && !text.isEmpty()) {
                        tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, "clover-s23-utt");
                    }
                }
            });
        }

        @JavascriptInterface
        public void stopSpeaking() {
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    if (tts != null) {
                        tts.stop();
                    }
                }
            });
        }

        @JavascriptInterface
        public void copyToClipboard(final String text) {
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    ClipboardManager cm = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
                    if (cm != null) {
                        cm.setPrimaryClip(ClipData.newPlainText("Clover AI Studio", text));
                        Toast.makeText(MainActivity.this, "Copied to clipboard", Toast.LENGTH_SHORT).show();
                    }
                }
            });
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (webView != null) {
            webView.post(new Runnable() {
                @Override
                public void run() {
                    webView.evaluateJavascript(
                            "if(window.k3RefreshOsPermissionBadges){window.k3RefreshOsPermissionBadges();}",
                            null
                    );
                }
            });
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQ_FILE_CHOOSER) {
            if (filePathCallback == null) return;
            Uri[] results = null;
            if (resultCode == Activity.RESULT_OK && data != null) {
                if (data.getClipData() != null) {
                    ClipData clipData = data.getClipData();
                    int count = clipData.getItemCount();
                    results = new Uri[count];
                    for (int i = 0; i < count; i++) {
                        results[i] = clipData.getItemAt(i).getUri();
                    }
                } else if (data.getData() != null) {
                    results = new Uri[]{data.getData()};
                }
            }
            filePathCallback.onReceiveValue(results);
            filePathCallback = null;
        } else if (requestCode == REQ_SPEECH_INPUT && resultCode == Activity.RESULT_OK && data != null) {
            ArrayList<String> matches = data.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS);
            if (matches != null && !matches.isEmpty() && webView != null) {
                final String spoken = matches.get(0).replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ");
                webView.post(new Runnable() {
                    @Override
                    public void run() {
                        webView.evaluateJavascript(
                                "if(window.k3OnNativeVoiceResult){window.k3OnNativeVoiceResult('" + spoken + "');}",
                                null
                        );
                    }
                });
            }
        }
    }

    @Override
    public void onBackPressed() {
        if (webView != null && webView.canGoBack()) {
            webView.goBack();
        } else {
            super.onBackPressed();
        }
    }

    @Override
    protected void onDestroy() {
        if (tts != null) {
            tts.stop();
            tts.shutdown();
        }
        super.onDestroy();
    }
}
