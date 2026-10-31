#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$APP_DIR/build"
ANDROID_JAR="/data/data/com.termux/files/usr/share/java/android.jar"
FRAMEWORK_RES="/system/framework/framework-res.apk"
KEYSTORE="$APP_DIR/clover_studio_release.keystore"
UNSIGNED_APK="$BUILD_DIR/CloverAIStudio-unsigned.apk"
ALIGNED_APK="$BUILD_DIR/CloverAIStudio-aligned.apk"
FINAL_APK="$APP_DIR/CloverAIStudio.apk"

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR/gen" "$BUILD_DIR/classes" "$BUILD_DIR/dex" "$BUILD_DIR/src/ai/clover/studio"

SECRET_FILE="$(dirname "$APP_DIR")/.app_secret"
if [ ! -f "$SECRET_FILE" ]; then
  python3 -c "import secrets; from pathlib import Path; p=Path('$SECRET_FILE'); p.write_text('s23-'+secrets.token_hex(32)+'\n'); p.chmod(0o600)"
fi
DEVICE_SECRET="$(tr -d '\r\n' < "$SECRET_FILE")"
sed "s/__CLOVER_S23_DEVICE_SECRET__/$DEVICE_SECRET/g" \
  "$APP_DIR/src/ai/clover/studio/MainActivity.java" \
  > "$BUILD_DIR/src/ai/clover/studio/MainActivity.java"

echo "[1/6] Compiling and linking Android resources with aapt2 + framework-res.apk..."
aapt2 compile --dir "$APP_DIR/res" -o "$BUILD_DIR/res.zip"
aapt2 link \
  -I "$FRAMEWORK_RES" \
  --manifest "$APP_DIR/AndroidManifest.xml" \
  -A "$APP_DIR/assets" \
  --java "$BUILD_DIR/gen" \
  -o "$UNSIGNED_APK" \
  "$BUILD_DIR/res.zip"

echo "[2/6] Compiling ai.clover.studio.MainActivity with javac (target 1.8) and per-device secret..."
javac -source 1.8 -target 1.8 \
  -bootclasspath "$ANDROID_JAR" \
  -classpath "$ANDROID_JAR" \
  -d "$BUILD_DIR/classes" \
  "$BUILD_DIR/gen/ai/clover/studio/R.java" \
  "$BUILD_DIR/src/ai/clover/studio/MainActivity.java"

echo "[3/6] Compiling classes to Dalvik bytecode (classes.dex) with d8..."
CLASS_FILES=$(find "$BUILD_DIR/classes" -name "*.class")
d8 --lib "$ANDROID_JAR" --min-api 24 --output "$BUILD_DIR/dex" $CLASS_FILES

echo "[4/6] Adding classes.dex into APK..."
(
  cd "$BUILD_DIR/dex"
  zip -u "$UNSIGNED_APK" classes.dex
)

echo "[5/6] Aligning APK with zipalign..."
zipalign -f -p 4 "$UNSIGNED_APK" "$ALIGNED_APK"

if [ ! -f "$KEYSTORE" ]; then
  echo "Generating signing keystore for ai.clover.studio..."
  keytool -genkeypair -v \
    -keystore "$KEYSTORE" \
    -alias cloverstudio \
    -keyalg RSA \
    -keysize 2048 \
    -validity 10000 \
    -storepass clover2026 \
    -keypass clover2026 \
    -dname "CN=Clover AI Studio, OU=Clover Framework, O=Clover AI, L=Bengaluru, S=KA, C=IN"
fi

echo "[6/6] Signing CloverAIStudio.apk (v1 + v2 + v3 schemes) with apksigner..."
apksigner sign \
  --ks "$KEYSTORE" \
  --ks-key-alias cloverstudio \
  --ks-pass pass:clover2026 \
  --key-pass pass:clover2026 \
  --out "$FINAL_APK" \
  "$ALIGNED_APK"

apksigner verify --verbose "$FINAL_APK"
ls -lh "$FINAL_APK"
echo "SUCCESS: Built signed $FINAL_APK"
