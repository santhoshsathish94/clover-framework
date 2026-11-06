using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.IO.Compression;
using System.Net;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading;
using System.Web.Script.Serialization;
using System.Windows.Forms;
using Microsoft.Win32;

[assembly: AssemblyTitle("Clover AI Studio Setup")]
[assembly: AssemblyDescription("Installer for Clover AI Studio")]
[assembly: AssemblyCompany("Santhosh Narayanan")]
[assembly: AssemblyProduct("Clover AI Studio")]
[assembly: AssemblyCopyright("Copyright (c) 2026 Santhosh Narayanan")]
[assembly: AssemblyVersion("5.0.0.0")]
[assembly: AssemblyFileVersion("5.0.0.0")]
[assembly: ComVisible(false)]

namespace CloverAIStudioSetup
{
    internal sealed class SetupState
    {
        public readonly object Sync = new object();
        public int Step = 0;
        public string StepText = "Starting";
        public double Plan = 0.0;
        public double Done = 0.0;
        public string Item = "";
        public string ItemState = "";
        public double ItemDone = 0.0;
        public double ItemTotal = 0.0;
        public int ItemIndex = 0;
        public int Items = 0;
        public string Detail = "";
        public string State = "running"; // running, done, failed, autoclose
        public string Message = "";
        public bool Cancel = false;
        public bool StartStudio = false;
        public Process ActiveProcess = null;
    }

    internal sealed class BufferedBar : Panel
    {
        public BufferedBar()
        {
            this.DoubleBuffered = true;
            this.SetStyle(ControlStyles.AllPaintingInWmPaint | ControlStyles.UserPaint | ControlStyles.OptimizedDoubleBuffer, true);
        }
    }

    internal static class Program
    {
        private const string VersionString = "0.5.0-beta.1";
        private static string LogPath;
        private static readonly object LogLock = new object();

        [DllImport("user32.dll")]
        private static extern bool SetProcessDPIAware();

        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Auto)]
        private class MEMORYSTATUSEX
        {
            public uint dwLength = (uint)Marshal.SizeOf(typeof(MEMORYSTATUSEX));
            public uint dwMemoryLoad;
            public ulong ullTotalPhys;
            public ulong ullAvailPhys;
            public ulong ullTotalPageFile;
            public ulong ullAvailPageFile;
            public ulong ullTotalVirtual;
            public ulong ullAvailVirtual;
            public ulong ullAvailExtendedVirtual;
        }

        [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool GlobalMemoryStatusEx([In, Out] MEMORYSTATUSEX lpBuffer);

        [DllImport("kernel32.dll")]
        private static extern bool IsProcessorFeaturePresent(uint processorFeature);

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool DeleteFileW(string lpFileName);

        private static void UnblockFile(string filePath)
        {
            if (string.IsNullOrEmpty(filePath)) return;
            try { DeleteFileW(filePath + ":Zone.Identifier"); } catch { }
        }

        private static void EnsureTrustedPublisher(bool includeRoot, string saveCerDir)
        {
            try
            {
                UnblockFile(Assembly.GetExecutingAssembly().Location);
            }
            catch { }
            try
            {
                using (Stream cs = Assembly.GetExecutingAssembly().GetManifestResourceStream("CloverAIStudio.codesign.cer"))
                {
                    if (cs == null) return;
                    using (MemoryStream ms = new MemoryStream())
                    {
                        cs.CopyTo(ms);
                        byte[] rawCert = ms.ToArray();
                        if (!string.IsNullOrEmpty(saveCerDir))
                        {
                            try
                            {
                                Directory.CreateDirectory(saveCerDir);
                                File.WriteAllBytes(Path.Combine(saveCerDir, "CloverFramework-CodeSign.cer"), rawCert);
                            }
                            catch { }
                        }
                        X509Certificate2 cert = new X509Certificate2(rawCert);
                        using (X509Store pubStore = new X509Store(StoreName.TrustedPublisher, StoreLocation.CurrentUser))
                        {
                            pubStore.Open(OpenFlags.ReadWrite);
                            if (!pubStore.Certificates.Contains(cert))
                            {
                                pubStore.Add(cert);
                            }
                        }
                        if (includeRoot)
                        {
                            using (X509Store rootStore = new X509Store(StoreName.Root, StoreLocation.CurrentUser))
                            {
                                rootStore.Open(OpenFlags.ReadWrite);
                                if (!rootStore.Certificates.Contains(cert))
                                {
                                    rootStore.Add(cert);
                                }
                            }
                        }
                    }
                }
            }
            catch { }
        }

        private static void UnblockDirectory(string dirPath)
        {
            if (string.IsNullOrEmpty(dirPath) || !Directory.Exists(dirPath)) return;
            try
            {
                foreach (string f in Directory.GetFiles(dirPath, "*", SearchOption.AllDirectories))
                {
                    UnblockFile(f);
                }
            }
            catch { }
        }

        [STAThread]
        private static int Main(string[] args)
        {
            try { SetProcessDPIAware(); } catch { }
            ServicePointManager.SecurityProtocol = SecurityProtocolType.Tls12;
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);

            bool quiet = false;
            bool noShortcuts = false;
            bool trustOnly = false;
            string baseDir = "";
            string components = "core,csi";
            string reuseDir = "";

            if (args != null)
            {
                for (int i = 0; i < args.Length; i++)
                {
                    string a = args[i] ?? "";
                    if (Eq(a, "-Quiet") || Eq(a, "--quiet") || Eq(a, "/Quiet") || Eq(a, "/Q")) quiet = true;
                    else if (Eq(a, "-NoShortcuts") || Eq(a, "--no-shortcuts")) noShortcuts = true;
                    else if (Eq(a, "-Trust") || Eq(a, "--trust")) trustOnly = true;
                    else if ((Eq(a, "-Base") || Eq(a, "--base")) && i + 1 < args.Length) baseDir = args[++i];
                    else if ((Eq(a, "-Components") || Eq(a, "--components")) && i + 1 < args.Length) components = args[++i];
                    else if ((Eq(a, "-Reuse") || Eq(a, "--reuse")) && i + 1 < args.Length) reuseDir = args[++i];
                }
            }

            EnsureTrustedPublisher(trustOnly, null);
            if (trustOnly)
            {
                return 0;
            }

            if (string.IsNullOrWhiteSpace(baseDir))
            {
                baseDir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "CloverAI");
            }
            baseDir = Path.GetFullPath(baseDir);
            LogPath = Path.Combine(Path.GetTempPath(), "CloverAIStudio-setup.log");
            Log(string.Format("setup {0} start, base={1} components={2} reuse={3} quiet={4}", VersionString, baseDir, components, reuseDir, quiet));

            SetupState state = new SetupState();

            if (quiet)
            {
                return RunInstallerWorker(state, baseDir, components, reuseDir, true, noShortcuts);
            }

            int exitCode = 0;
            Thread worker = new Thread(() =>
            {
                exitCode = RunInstallerWorker(state, baseDir, components, reuseDir, false, noShortcuts);
            });
            worker.IsBackground = true;
            worker.Start();

            RunSetupWindow(state);
            worker.Join(5000);

            if (exitCode == 0 || state.StartStudio)
            {
                string studioExe = Path.Combine(baseDir, "CloverAIStudio.exe");
                if (File.Exists(studioExe) && !state.Cancel)
                {
                    try
                    {
                        Process.Start(new ProcessStartInfo(studioExe) { WorkingDirectory = baseDir, UseShellExecute = false });
                    }
                    catch { }
                }
            }
            return exitCode;
        }

        private static bool Eq(string a, string b)
        {
            return string.Equals(a, b, StringComparison.OrdinalIgnoreCase);
        }

        private static void Log(string msg)
        {
            try
            {
                lock (LogLock)
                {
                    File.AppendAllText(LogPath, DateTime.Now.ToString("s") + " " + msg + Environment.NewLine, Encoding.UTF8);
                }
            }
            catch { }
        }

        private static void SetStep(SetupState st, string text, int stepNum)
        {
            Log("== " + text);
            lock (st.Sync)
            {
                if (st.Cancel) throw new OperationCanceledException("stopped");
                st.Step = stepNum;
                st.StepText = text;
                st.Item = "";
                st.ItemDone = 0.0;
                st.ItemTotal = 0.0;
                st.Detail = "";
            }
        }

        private static int RunInstallerWorker(SetupState st, string baseDir, string components, string reuseDir, bool quiet, bool noShortcuts)
        {
            try
            {
                // 1. Check computer
                SetStep(st, "Checking this computer", 1);
                Directory.CreateDirectory(baseDir);
                MEMORYSTATUSEX mem = new MEMORYSTATUSEX();
                double ramGb = GlobalMemoryStatusEx(mem) ? Math.Round(mem.ullTotalPhys / (1024.0 * 1024.0 * 1024.0), 1) : 16.0;
                int threads = Environment.ProcessorCount;
                int build = GetWindowsBuildNumber();
                bool avx2 = false;
                try { avx2 = IsProcessorFeaturePresent(40); } catch { avx2 = true; }
                string rootDrive = Path.GetPathRoot(baseDir);
                double freeGb = Math.Round(new DriveInfo(rootDrive).AvailableFreeSpace / (1024.0 * 1024.0 * 1024.0), 1);
                bool existingVenv = File.Exists(Path.Combine(baseDir, @"venv\Scripts\python.exe"));

                List<string> low = new List<string>();
                if (!Environment.Is64BitOperatingSystem) low.Add("a 64-bit Intel or AMD processor");
                if (build < 19041) low.Add("Windows 10 version 2004 or later (this is build " + build + ")");
                if (ramGb < 15.0) low.Add("16 GB of memory (this computer has " + ramGb + " GB)");
                if (threads < 8) low.Add("8 processor threads (this computer has " + threads + ")");
                if (!avx2) low.Add("a processor with AVX2");
                int needGb = existingVenv ? 8 : 25;
                if (freeGb < needGb) low.Add(needGb + " GB free on the drive of " + baseDir + " (" + freeGb + " GB free)");

                Log(string.Format("{0} threads, AVX2 {1}, {2} GB memory, Windows build {3}, {4} GB free", threads, avx2, ramGb, build, freeGb));
                if (low.Count > 0)
                {
                    FinishState(st, quiet, "failed", "Clover AI Studio needs at least the computer it was built on. This one is missing:\n\n- " + string.Join("\n- ", low));
                    return 2;
                }

                if (!IsVcRedistInstalled())
                {
                    SetStep(st, "Installing the Microsoft Visual C++ runtime", 1);
                    string vcExe = Path.Combine(Path.GetTempPath(), "vc_redist.x64.exe");
                    DownloadFileWithProgress(st, "https://aka.ms/vs/17/release/vc_redist.x64.exe", vcExe, null);
                    ProcessStartInfo vcPsi = new ProcessStartInfo(vcExe, "/install /quiet /norestart")
                    {
                        UseShellExecute = true,
                        Verb = "runas"
                    };
                    using (Process p = Process.Start(vcPsi))
                    {
                        if (p != null) p.WaitForExit();
                        int rc = p != null ? p.ExitCode : -1;
                        Log("vc_redist exit " + rc);
                        if (rc != 0 && rc != 1638 && rc != 3010)
                        {
                            throw new Exception("The Visual C++ runtime did not install (exit " + rc + ")");
                        }
                    }
                }

                // 2. Install the app
                SetStep(st, "Installing the app", 2);
                TryShutdownRunningEngine();
                StopRunningLauncher(baseDir);

                string cloverPy = Path.Combine(baseDir, @"clover\server.py");
                if (File.Exists(cloverPy))
                {
                    string stamp = DateTime.Now.ToString("yyyyMMdd-HHmmss");
                    string backupDir = Path.Combine(baseDir, "backup");
                    Directory.CreateDirectory(backupDir);
                    string bkZip = Path.Combine(backupDir, "clover-" + stamp + ".zip");
                    try
                    {
                        if (File.Exists(bkZip)) File.Delete(bkZip);
                        ZipFile.CreateFromDirectory(Path.Combine(baseDir, "clover"), bkZip);
                        DirectoryInfo di = new DirectoryInfo(backupDir);
                        FileInfo[] bks = di.GetFiles("clover-*.zip");
                        Array.Sort(bks, (x, y) => y.LastWriteTimeUtc.CompareTo(x.LastWriteTimeUtc));
                        for (int i = 2; i < bks.Length; i++)
                        {
                            try { bks[i].Delete(); } catch { }
                        }
                    }
                    catch (Exception ex) { Log("backup note: " + ex.Message); }
                }

                string newDir = Path.Combine(baseDir, "setup-new");
                RemoveTree(newDir);
                Directory.CreateDirectory(newDir);

                using (Stream res = Assembly.GetExecutingAssembly().GetManifestResourceStream("CloverAIStudio.app.zip"))
                {
                    if (res == null) throw new Exception("Embedded resource CloverAIStudio.app.zip was not found.");
                    string tmpZip = Path.Combine(newDir, "_payload.zip");
                    using (FileStream fs = File.Create(tmpZip))
                    {
                        res.CopyTo(fs);
                    }
                    ZipFile.ExtractToDirectory(tmpZip, newDir);
                    File.Delete(tmpZip);
                }

                foreach (string sub in new[] { "clover", "studio" })
                {
                    string dstSub = Path.Combine(baseDir, sub);
                    string srcSub = Path.Combine(newDir, sub);
                    RemoveTree(dstSub);
                    if (Directory.Exists(srcSub)) Directory.Move(srcSub, dstSub);
                }
                foreach (string f in Directory.GetFiles(newDir))
                {
                    File.Copy(f, Path.Combine(baseDir, Path.GetFileName(f)), true);
                }
                RemoveTree(newDir);
                EnsureTrustedPublisher(false, Path.Combine(baseDir, "studio"));
                UnblockDirectory(baseDir);

                JavaScriptSerializer jss = new JavaScriptSerializer { MaxJsonLength = int.MaxValue };
                string manJson = File.ReadAllText(Path.Combine(baseDir, @"clover\launch\manifest.json"), Encoding.UTF8);
                Dictionary<string, object> man = jss.Deserialize<Dictionary<string, object>>(manJson);
                Dictionary<string, object> uvSpec = (Dictionary<string, object>)man["uv"];
                string uvVersion = Convert.ToString(uvSpec["version"]);
                string uvUrl = Convert.ToString(uvSpec["url"]);
                string uvSha = Convert.ToString(uvSpec["sha256"]);
                string pyVersion = Convert.ToString(man["python"]);

                // 3. uv, Python, packages
                SetStep(st, "uv " + uvVersion + " (installs Python and the packages)", 3);
                string uvDir = Path.Combine(baseDir, @"tools\uv");
                string uvExe = Path.Combine(uvDir, "uv.exe");
                bool haveUv = false;
                if (File.Exists(uvExe))
                {
                    string verOut = RunCapture(uvExe, "--version", baseDir);
                    haveUv = verOut.IndexOf(uvVersion, StringComparison.OrdinalIgnoreCase) >= 0;
                }
                if (!haveUv)
                {
                    string dlDir = Path.Combine(baseDir, "downloads");
                    Directory.CreateDirectory(dlDir);
                    Directory.CreateDirectory(uvDir);
                    string uvZip = Path.Combine(dlDir, "uv-x86_64-pc-windows-msvc.zip");
                    if (!File.Exists(uvZip) || !Eq(ComputeSha256(uvZip), uvSha))
                    {
                        DownloadFileWithProgress(st, uvUrl, uvZip, uvSha);
                    }
                    RemoveTree(uvDir);
                    Directory.CreateDirectory(uvDir);
                    ZipFile.ExtractToDirectory(uvZip, uvDir);
                    try { File.Delete(uvZip); } catch { }
                }

                Dictionary<string, string> uvEnv = new Dictionary<string, string>
                {
                    { "UV_PYTHON_INSTALL_DIR", Path.Combine(baseDir, @"runtime\python") },
                    { "UV_CACHE_DIR", Path.Combine(baseDir, @"runtime\uv-cache") },
                    { "UV_NO_CONFIG", "1" },
                    { "UV_PYTHON_DOWNLOADS", "automatic" },
                    { "PYTHONIOENCODING", "utf-8" }
                };

                SetStep(st, "Python " + pyVersion, 3);
                RunTool(st, uvExe, "python install --no-bin --no-registry " + pyVersion, baseDir, uvEnv, "installing Python");

                string pyExe = Path.Combine(baseDir, @"venv\Scripts\python.exe");
                string curPyVer = File.Exists(pyExe)
                    ? RunCapture(pyExe, "-c \"import sys; print('.'.join(map(str, sys.version_info[:3])))\"", baseDir).Trim()
                    : "";
                if (!Eq(curPyVer, pyVersion))
                {
                    RemoveTree(Path.Combine(baseDir, "venv"));
                    RunTool(st, uvExe, string.Format("venv --python {0} \"{1}\"", pyVersion, Path.Combine(baseDir, "venv")), baseDir, uvEnv, "making the Python environment");
                }

                SetStep(st, "Python packages (locked; each file checked against its SHA-256)", 3);
                string reqLock = Path.Combine(baseDir, @"studio\requirements.lock");
                string reqStamp = Path.Combine(baseDir, @"config\requirements.sha256");
                string reqHash = ComputeSha256(reqLock);
                bool havePkgs = File.Exists(reqStamp) && Eq(File.ReadAllText(reqStamp).Trim(), reqHash);
                if (!havePkgs)
                {
                    string impOk = RunCapture(pyExe, "-c \"import numpy, openpyxl, pypdf, docx, pptx, PIL; print('OK')\"", baseDir).Trim();
                    if (impOk == "OK")
                    {
                        Directory.CreateDirectory(Path.GetDirectoryName(reqStamp));
                        File.WriteAllText(reqStamp, reqHash, Encoding.ASCII);
                        havePkgs = true;
                    }
                }
                if (!havePkgs)
                {
                    RunTool(st, uvExe, string.Format("pip sync --python \"{0}\" --require-hashes \"{1}\"", pyExe, reqLock), baseDir, uvEnv, "installing the Python packages");
                    Directory.CreateDirectory(Path.GetDirectoryName(reqStamp));
                    File.WriteAllText(reqStamp, reqHash, Encoding.ASCII);
                    try { RunCapture(uvExe, "cache clean", baseDir); } catch { }
                }

                // 4. Models, tools and data
                SetStep(st, "Models, tools and data", 4);
                if (quiet)
                {
                    string fetchScript = Path.Combine(baseDir, @"clover\setup_fetch.py");
                    string fetchArgs = string.Format("\"{0}\" --base \"{1}\" --components \"{2}\"", fetchScript, baseDir, components);
                    if (!string.IsNullOrEmpty(reuseDir)) fetchArgs += string.Format(" --reuse \"{0}\"", reuseDir);
                    RunTool(st, pyExe, fetchArgs, baseDir, uvEnv, "getting the models and tools");
                }
                else
                {
                    string gitExe = Path.Combine(baseDir, @"tools\git\cmd\git.exe");
                    if (!File.Exists(gitExe) && man.ContainsKey("archives"))
                    {
                        ArrayList archives = man["archives"] as ArrayList;
                        if (archives != null)
                        {
                            foreach (object itemObj in archives)
                            {
                                Dictionary<string, object> item = itemObj as Dictionary<string, object>;
                                if (item != null && Eq(Convert.ToString(item["dest"]), "tools/git"))
                                {
                                    string gUrl = Convert.ToString(item["url"]);
                                    string gSha = Convert.ToString(item["sha256"]);
                                    string gZip = Path.Combine(baseDir, @"downloads\MinGit.zip");
                                    string gDest = Path.Combine(baseDir, @"tools\git");
                                    Directory.CreateDirectory(Path.GetDirectoryName(gZip));
                                    DownloadFileWithProgress(st, gUrl, gZip, gSha);
                                    RemoveTree(gDest);
                                    Directory.CreateDirectory(gDest);
                                    ZipFile.ExtractToDirectory(gZip, gDest);
                                    File.WriteAllText(Path.Combine(gDest, ".clover-archive"), gSha, Encoding.ASCII);
                                    try { File.Delete(gZip); } catch { }
                                    break;
                                }
                            }
                        }
                    }
                    Directory.CreateDirectory(Path.Combine(baseDir, "config"));
                    File.WriteAllText(Path.Combine(baseDir, @"config\setup-walkthrough.flag"), VersionString, Encoding.ASCII);
                }

                // 5. Hardware
                SetStep(st, "Reading the hardware", 5);
                string hwOut = RunCapture(pyExe, string.Format("\"{0}\" \"{1}\"", Path.Combine(baseDir, @"clover\hardware.py"), baseDir), baseDir);
                Log(hwOut);

                // 6. Shortcuts & VS Code
                if (!noShortcuts)
                {
                    SetStep(st, "Shortcuts and the VS Code extension", 6);
                    string studioExe = Path.Combine(baseDir, "CloverAIStudio.exe");
                    string programs = Environment.GetFolderPath(Environment.SpecialFolder.Programs);
                    string desktop = Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory);
                    CreateShortcut(Path.Combine(programs, "Clover AI Studio.lnk"), studioExe, "", baseDir, "Clover AI Studio (on this computer)");
                    CreateShortcut(Path.Combine(programs, "Clover Engine for VS Code.lnk"), studioExe, "--engine-only", baseDir, "Start the Clover engine for VS Code");
                    CreateShortcut(Path.Combine(desktop, "Clover AI Studio.lnk"), studioExe, "", baseDir, "Clover AI Studio (on this computer)");

                    string[] codePaths = {
                        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), @"Programs\Microsoft VS Code\bin\code.cmd"),
                        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles), @"Microsoft VS Code\bin\code.cmd")
                    };
                    string codeCmd = null;
                    foreach (string cp in codePaths) { if (File.Exists(cp)) { codeCmd = cp; break; } }
                    string studioFolder = Path.Combine(baseDir, "studio");
                    string[] vsixFiles = Directory.Exists(studioFolder) ? Directory.GetFiles(studioFolder, "clover-local-*.vsix") : new string[0];
                    if (codeCmd != null && vsixFiles.Length > 0)
                    {
                        Array.Sort(vsixFiles, StringComparer.OrdinalIgnoreCase);
                        string vsix = vsixFiles[vsixFiles.Length - 1];
                        string vOut = RunCapture(codeCmd, string.Format("--install-extension \"{0}\" --force", vsix), baseDir);
                        Log("code --install-extension: " + vOut);
                    }

                    using (RegistryKey uk = Registry.CurrentUser.CreateSubKey(@"Software\Microsoft\Windows\CurrentVersion\Uninstall\CloverAIStudio"))
                    {
                        if (uk != null)
                        {
                            uk.SetValue("DisplayName", "Clover AI Studio");
                            uk.SetValue("DisplayVersion", VersionString);
                            uk.SetValue("Publisher", "Santhosh Narayanan");
                            uk.SetValue("DisplayIcon", studioExe + ",0");
                            uk.SetValue("InstallLocation", baseDir);
                            uk.SetValue("HelpLink", "https://github.com/santhoshsathish94/clover-framework/issues");
                            uk.SetValue("URLInfoAbout", "https://cloverframework.com/");
                            uk.SetValue("NoModify", 1, RegistryValueKind.DWord);
                            uk.SetValue("NoRepair", 1, RegistryValueKind.DWord);
                            uk.SetValue("UninstallString", "\"" + studioExe + "\" --uninstall");
                        }
                    }
                }

                Log("SETUP_DONE");
                if (!quiet)
                {
                    lock (st.Sync)
                    {
                        st.StartStudio = true;
                        st.State = "autoclose";
                    }
                }
                else
                {
                    Console.WriteLine("Clover AI Studio " + VersionString + " is installed in " + baseDir + ".");
                }
                return 0;
            }
            catch (OperationCanceledException)
            {
                Log("SETUP_STOPPED by the user");
                FinishState(st, quiet, "failed", "Setup was stopped. Nothing is lost: run setup again and it continues where it stopped.");
                return 3;
            }
            catch (Exception ex)
            {
                Log("SETUP_FAILED " + ex);
                FinishState(st, quiet, "failed", "Setup failed: " + ex.Message + "\n\nDetails: " + LogPath + "\n\nRun setup again to continue where it stopped.");
                return 1;
            }
        }

        private static void FinishState(SetupState st, bool quiet, string state, string msg)
        {
            Log(msg);
            if (quiet)
            {
                Console.WriteLine(msg);
                return;
            }
            lock (st.Sync)
            {
                st.Message = msg;
                st.State = state;
            }
        }

        private static int GetWindowsBuildNumber()
        {
            try
            {
                using (RegistryKey k = Registry.LocalMachine.OpenSubKey(@"SOFTWARE\Microsoft\Windows NT\CurrentVersion"))
                {
                    if (k != null)
                    {
                        string s = Convert.ToString(k.GetValue("CurrentBuildNumber") ?? k.GetValue("CurrentBuild"));
                        int b;
                        if (int.TryParse(s, out b)) return b;
                    }
                }
            }
            catch { }
            return Environment.OSVersion.Version.Build;
        }

        private static bool IsVcRedistInstalled()
        {
            try
            {
                using (RegistryKey k = Registry.LocalMachine.OpenSubKey(@"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64"))
                {
                    if (k != null)
                    {
                        object v = k.GetValue("Installed");
                        if (v != null && Convert.ToInt32(v) == 1) return true;
                    }
                }
            }
            catch { }
            return false;
        }

        private static void TryShutdownRunningEngine()
        {
            try
            {
                HttpWebRequest req = (HttpWebRequest)WebRequest.Create("http://127.0.0.1:8765/api/shutdown");
                req.Method = "POST";
                req.ContentType = "application/json";
                req.Timeout = 3000;
                byte[] b = Encoding.UTF8.GetBytes("{}");
                req.ContentLength = b.Length;
                using (Stream s = req.GetRequestStream()) s.Write(b, 0, b.Length);
                using (WebResponse r = req.GetResponse()) { }
                Thread.Sleep(2000);
            }
            catch { }
        }

        private static void StopRunningLauncher(string baseDir)
        {
            string target = Path.Combine(baseDir, "CloverAIStudio.exe");
            foreach (Process p in Process.GetProcessesByName("CloverAIStudio"))
            {
                try
                {
                    if (p.Id != Process.GetCurrentProcess().Id &&
                        string.Equals(p.MainModule.FileName, target, StringComparison.OrdinalIgnoreCase))
                    {
                        p.Kill();
                        p.WaitForExit(3000);
                    }
                }
                catch { }
            }
        }

        private static void RemoveTree(string path)
        {
            for (int i = 0; i < 3 && Directory.Exists(path); i++)
            {
                if (i > 0) Thread.Sleep(1500);
                try
                {
                    foreach (string f in Directory.GetFiles(path, "*", SearchOption.AllDirectories))
                    {
                        try { File.SetAttributes(f, FileAttributes.Normal); } catch { }
                    }
                    Directory.Delete(path, true);
                }
                catch { }
            }
        }

        private static string ComputeSha256(string filePath)
        {
            using (SHA256 sha = SHA256.Create())
            using (FileStream fs = File.OpenRead(filePath))
            {
                byte[] hash = sha.ComputeHash(fs);
                StringBuilder sb = new StringBuilder(hash.Length * 2);
                foreach (byte b in hash) sb.Append(b.ToString("x2"));
                return sb.ToString();
            }
        }

        private static void DownloadFileWithProgress(SetupState st, string url, string destPath, string expectedSha256)
        {
            HttpWebRequest req = (HttpWebRequest)WebRequest.Create(url);
            req.UserAgent = "CloverAIStudioSetup/" + VersionString;
            req.Timeout = 60000;
            req.ReadWriteTimeout = 60000;
            using (HttpWebResponse resp = (HttpWebResponse)req.GetResponse())
            using (Stream rs = resp.GetResponseStream())
            using (FileStream fs = File.Create(destPath))
            {
                long total = resp.ContentLength;
                byte[] buf = new byte[65536];
                long done = 0;
                int n;
                while ((n = rs.Read(buf, 0, buf.Length)) > 0)
                {
                    lock (st.Sync)
                    {
                        if (st.Cancel) throw new OperationCanceledException("stopped");
                        st.ItemDone = done;
                        if (total > 0) st.ItemTotal = total;
                    }
                    fs.Write(buf, 0, n);
                    done += n;
                }
            }
            if (!string.IsNullOrEmpty(expectedSha256))
            {
                string actual = ComputeSha256(destPath);
                if (!Eq(actual, expectedSha256))
                {
                    try { File.Delete(destPath); } catch { }
                    throw new Exception(Path.GetFileName(destPath) + " does not match its published SHA-256");
                }
            }
        }

        private static string RunCapture(string exe, string arguments, string workDir)
        {
            try
            {
                ProcessStartInfo psi = new ProcessStartInfo(exe, arguments)
                {
                    WorkingDirectory = workDir,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true
                };
                using (Process p = Process.Start(psi))
                {
                    string stdout = p.StandardOutput.ReadToEnd();
                    string stderr = p.StandardError.ReadToEnd();
                    p.WaitForExit(120000);
                    return stdout + "\n" + stderr;
                }
            }
            catch
            {
                return "";
            }
        }

        private static void RunTool(SetupState st, string exe, string arguments, string workDir, Dictionary<string, string> env, string what)
        {
            ProcessStartInfo psi = new ProcessStartInfo(exe, arguments)
            {
                WorkingDirectory = workDir,
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true
            };
            if (env != null)
            {
                foreach (KeyValuePair<string, string> kv in env)
                {
                    psi.EnvironmentVariables[kv.Key] = kv.Value;
                }
            }
            JavaScriptSerializer jss = new JavaScriptSerializer();
            using (Process p = new Process())
            {
                p.StartInfo = psi;
                DataReceivedEventHandler handler = (sender, e) =>
                {
                    if (string.IsNullOrEmpty(e.Data)) return;
                    string s = e.Data;
                    if (s.StartsWith("@@"))
                    {
                        ParseProgressEvent(st, jss, s.Substring(2));
                        return;
                    }
                    Log(s);
                    lock (st.Sync)
                    {
                        if (!string.IsNullOrWhiteSpace(s)) st.Detail = s.Trim();
                    }
                };
                p.OutputDataReceived += handler;
                p.ErrorDataReceived += handler;
                p.Start();
                lock (st.Sync) { st.ActiveProcess = p; }
                p.BeginOutputReadLine();
                p.BeginErrorReadLine();
                p.WaitForExit();
                lock (st.Sync) { st.ActiveProcess = null; }
                if (st.Cancel) throw new OperationCanceledException("stopped");
                if (p.ExitCode != 0)
                {
                    throw new Exception(what + " failed (exit " + p.ExitCode + "); see " + LogPath);
                }
            }
        }

        private static void ParseProgressEvent(SetupState st, JavaScriptSerializer jss, string json)
        {
            try
            {
                Dictionary<string, object> e = jss.Deserialize<Dictionary<string, object>>(json);
                if (e == null || !e.ContainsKey("ev")) return;
                string ev = Convert.ToString(e["ev"]);
                lock (st.Sync)
                {
                    if (ev == "plan")
                    {
                        st.Plan = Convert.ToDouble(e["total"]);
                        st.Done = 0.0;
                    }
                    else if (ev == "item")
                    {
                        string istate = Convert.ToString(e["state"]);
                        if (istate == "checking")
                        {
                            st.Item = Convert.ToString(e["name"]);
                            st.ItemState = "checking";
                            st.ItemDone = 0.0;
                            st.ItemTotal = e.ContainsKey("size") && e["size"] != null ? Convert.ToDouble(e["size"]) : 0.0;
                            st.ItemIndex = e.ContainsKey("i") && e["i"] != null ? Convert.ToInt32(e["i"]) : 0;
                            st.Items = e.ContainsKey("n") && e["n"] != null ? Convert.ToInt32(e["n"]) : 0;
                        }
                        else if (istate == "getting")
                        {
                            st.ItemState = "getting";
                            if (e.ContainsKey("size") && e["size"] != null) st.ItemTotal = Convert.ToDouble(e["size"]);
                        }
                        else
                        {
                            st.Done += st.ItemTotal;
                            st.ItemDone = 0.0;
                            st.ItemState = istate;
                        }
                    }
                    else if (ev == "bytes")
                    {
                        st.ItemDone = Convert.ToDouble(e["done"]);
                        if (e.ContainsKey("total") && e["total"] != null) st.ItemTotal = Convert.ToDouble(e["total"]);
                    }
                }
            }
            catch { }
        }

        private static void CreateShortcut(string lnkPath, string targetPath, string arguments, string workDir, string description)
        {
            try
            {
                Type t = Type.GetTypeFromProgID("WScript.Shell");
                if (t == null) return;
                object shell = Activator.CreateInstance(t);
                object sc = t.InvokeMember("CreateShortcut", BindingFlags.InvokeMethod, null, shell, new object[] { lnkPath });
                Type st = sc.GetType();
                st.InvokeMember("TargetPath", BindingFlags.SetProperty, null, sc, new object[] { targetPath });
                st.InvokeMember("Arguments", BindingFlags.SetProperty, null, sc, new object[] { arguments });
                st.InvokeMember("WorkingDirectory", BindingFlags.SetProperty, null, sc, new object[] { workDir });
                st.InvokeMember("IconLocation", BindingFlags.SetProperty, null, sc, new object[] { targetPath + ",0" });
                st.InvokeMember("Description", BindingFlags.SetProperty, null, sc, new object[] { description });
                st.InvokeMember("Save", BindingFlags.InvokeMethod, null, sc, null);
            }
            catch (Exception ex)
            {
                Log("shortcut note: " + ex.Message);
            }
        }

        private static void RunSetupWindow(SetupState st)
        {
            Color cream = ColorTranslator.FromHtml("#F7F3E9");
            Color surface = ColorTranslator.FromHtml("#FFFDF8");
            Color fill = ColorTranslator.FromHtml("#ECE6D7");
            Color green = ColorTranslator.FromHtml("#176B2C");
            Color ink = ColorTranslator.FromHtml("#1B271F");
            Color ink2 = ColorTranslator.FromHtml("#39473D");
            Color muted = ColorTranslator.FromHtml("#687469");
            Color faint = ColorTranslator.FromHtml("#98A198");
            Color warn = ColorTranslator.FromHtml("#85540F");

            Form f = new Form
            {
                Text = "Clover AI Studio Setup",
                ClientSize = new Size(600, 470),
                StartPosition = FormStartPosition.CenterScreen,
                FormBorderStyle = FormBorderStyle.FixedSingle,
                MaximizeBox = false,
                BackColor = cream,
                ForeColor = ink,
                Font = new Font("Segoe UI", 10f),
                AutoScaleDimensions = new SizeF(96f, 96f),
                AutoScaleMode = AutoScaleMode.Dpi
            };

            try
            {
                Icon appIcon = Icon.ExtractAssociatedIcon(Assembly.GetExecutingAssembly().Location);
                if (appIcon != null)
                {
                    f.Icon = appIcon;
                    PictureBox pb = new PictureBox
                    {
                        Image = appIcon.ToBitmap(),
                        SizeMode = PictureBoxSizeMode.Zoom,
                        Location = new Point(32, 30),
                        Size = new Size(44, 44)
                    };
                    f.Controls.Add(pb);
                }
            }
            catch { }

            Func<string, int, int, int, int, float, string, Color, Label> makeLbl = (txt, x, y, w, h, sz, face, col) =>
            {
                Label l = new Label
                {
                    Text = txt,
                    Location = new Point(x, y),
                    Size = new Size(w, h),
                    Font = new Font(face, sz),
                    ForeColor = col,
                    AutoEllipsis = true
                };
                f.Controls.Add(l);
                return l;
            };

            Label title = makeLbl("Installing Clover AI Studio", 90, 26, 480, 32, 15f, "Segoe UI Semibold", ink);
            makeLbl("Version " + VersionString + ". Everything comes from its publisher and is checked before it is used.", 90, 60, 480, 40, 9.5f, "Segoe UI", muted);

            string[] stepNames = {
                "Checking this computer",
                "Installing the app",
                "Python and its packages",
                "Models, tools and data",
                "Reading the hardware",
                "Shortcuts and VS Code"
            };
            Label[] rowMarks = new Label[stepNames.Length];
            Label[] rowTexts = new Label[stepNames.Length];
            string[] rowStates = new string[stepNames.Length];

            for (int i = 0; i < stepNames.Length; i++)
            {
                rowMarks[i] = makeLbl("\u25CB", 36, 114 + 28 * i, 24, 26, 11f, "Segoe UI Symbol", faint);
                rowTexts[i] = makeLbl(stepNames[i], 62, 114 + 28 * i, 500, 26, 10.5f, "Segoe UI", faint);
                rowStates[i] = "";
            }

            Label overall = makeLbl("Starting", 32, 292, 330, 24, 10f, "Segoe UI Semibold", ink2);
            Label overallR = makeLbl("", 362, 292, 206, 24, 10f, "Segoe UI Semibold", ink2);
            overallR.TextAlign = ContentAlignment.TopRight;
            Label msgBox = makeLbl("", 32, 292, 536, 116, 10f, "Segoe UI", ink2);
            msgBox.Visible = false;
            msgBox.AutoEllipsis = false;

            BufferedBar bar1 = new BufferedBar { Location = new Point(32, 318), Size = new Size(536, 10), BackColor = fill };
            BufferedBar bar2 = new BufferedBar { Location = new Point(32, 362), Size = new Size(536, 6), BackColor = fill };
            f.Controls.Add(bar1);
            f.Controls.Add(bar2);

            Label itemLbl = makeLbl("", 32, 336, 380, 24, 9.5f, "Segoe UI", ink2);
            Label itemR = makeLbl("", 412, 336, 156, 24, 9.5f, "Segoe UI", ink2);
            itemR.TextAlign = ContentAlignment.TopRight;
            Label detail = makeLbl("", 32, 374, 300, 22, 9f, "Segoe UI", muted);
            Label detailR = makeLbl("", 332, 374, 236, 22, 9f, "Segoe UI", muted);
            detailR.TextAlign = ContentAlignment.TopRight;

            Button btn1 = new Button { Text = "Cancel", Location = new Point(388, 414), Size = new Size(180, 36) };
            Button btn2 = new Button { Text = "Show details", Location = new Point(236, 414), Size = new Size(140, 36), Visible = false };
            foreach (Button b in new[] { btn1, btn2 })
            {
                b.FlatStyle = FlatStyle.Flat;
                b.FlatAppearance.BorderColor = fill;
                b.BackColor = surface;
                b.ForeColor = ink;
                b.Font = new Font("Segoe UI Semibold", 10f);
                b.Cursor = Cursors.Hand;
                f.Controls.Add(b);
            }

            double p1 = 0.0, p2 = 0.0, marquee = 0.0;
            bool indet = true;
            string lastItem = null;
            string lastState = "running";

            bar1.Paint += (s, e) =>
            {
                int w = (int)(bar1.Width * Math.Min(1.0, Math.Max(0.0, p1)));
                if (w > 0) using (SolidBrush br = new SolidBrush(green)) e.Graphics.FillRectangle(br, 0, 0, w, bar1.Height);
            };
            bar2.Paint += (s, e) =>
            {
                using (SolidBrush br = new SolidBrush(green))
                {
                    if (indet)
                    {
                        int bw = (int)(bar2.Width * 0.24);
                        int x = (int)((bar2.Width + bw) * marquee) - bw;
                        e.Graphics.FillRectangle(br, x, 0, bw, bar2.Height);
                    }
                    else
                    {
                        int w = (int)(bar2.Width * Math.Min(1.0, Math.Max(0.0, p2)));
                        if (w > 0) e.Graphics.FillRectangle(br, 0, 0, w, bar2.Height);
                    }
                }
            };

            System.Windows.Forms.Timer timer = new System.Windows.Forms.Timer { Interval = 50 };
            timer.Tick += (s, e) =>
            {
                int step;
                string curState, stepText, itemName, itemState, detailText, messageText;
                double plan, done, itemDone, itemTotal;
                int itemIdx, itemsCount;
                lock (st.Sync)
                {
                    step = st.Step;
                    curState = st.State;
                    stepText = st.StepText;
                    plan = st.Plan;
                    done = st.Done;
                    itemName = st.Item;
                    itemState = st.ItemState;
                    itemDone = st.ItemDone;
                    itemTotal = st.ItemTotal;
                    itemIdx = st.ItemIndex;
                    itemsCount = st.Items;
                    detailText = st.Detail;
                    messageText = st.Message;
                }

                for (int i = 0; i < stepNames.Length; i++)
                {
                    int n = i + 1;
                    string k = (curState == "done" || n < step) ? "done" : (n == step ? (curState == "running" ? "now" : "stop") : "todo");
                    if (rowStates[i] == k) continue;
                    rowStates[i] = k;
                    if (k == "done")
                    {
                        rowMarks[i].Text = "\u2713"; rowMarks[i].ForeColor = green;
                        rowTexts[i].ForeColor = ink2; rowTexts[i].Font = new Font("Segoe UI", 10.5f);
                    }
                    else if (k == "now")
                    {
                        rowMarks[i].Text = "\u25CF"; rowMarks[i].ForeColor = green;
                        rowTexts[i].ForeColor = ink; rowTexts[i].Font = new Font("Segoe UI Semibold", 10.5f);
                    }
                    else if (k == "stop")
                    {
                        rowMarks[i].Text = "\u25CF"; rowMarks[i].ForeColor = warn;
                        rowTexts[i].ForeColor = warn; rowTexts[i].Font = new Font("Segoe UI Semibold", 10.5f);
                    }
                    else
                    {
                        rowMarks[i].Text = "\u25CB"; rowMarks[i].ForeColor = faint;
                        rowTexts[i].ForeColor = faint; rowTexts[i].Font = new Font("Segoe UI", 10.5f);
                    }
                }

                if (curState != "running")
                {
                    if (curState == "autoclose")
                    {
                        timer.Stop();
                        f.Close();
                        return;
                    }
                    if (lastState != curState)
                    {
                        lastState = curState;
                        overall.Visible = overallR.Visible = itemLbl.Visible = itemR.Visible = detail.Visible = detailR.Visible = bar1.Visible = bar2.Visible = false;
                        msgBox.Text = messageText;
                        msgBox.Visible = true;
                        btn2.Visible = true;
                        btn1.Enabled = true;
                        if (curState == "done")
                        {
                            title.Text = "Clover AI Studio is ready";
                            btn1.Text = "Start Clover AI Studio";
                            btn1.BackColor = green;
                            btn1.ForeColor = Color.White;
                            btn1.FlatAppearance.BorderColor = green;
                            btn2.Text = "Close";
                        }
                        else
                        {
                            title.Text = "Setup stopped";
                            title.ForeColor = warn;
                            btn1.Text = "Close";
                            btn2.Text = "Show details";
                        }
                    }
                    return;
                }

                double cur = done + itemDone;
                double frac = plan > 0 ? Math.Min(1.0, cur / plan) : 0.0;
                double target = step <= 3 ? 0.08 * step : (step == 4 ? 0.25 + 0.65 * frac : 0.90 + 0.05 * (step - 4));
                p1 += (target - p1) * 0.15;
                bar1.Invalidate();

                overall.Text = stepText;
                if (step == 4 && !string.IsNullOrEmpty(itemName))
                {
                    if (lastItem != itemName) { lastItem = itemName; p2 = 0.0; }
                    if (itemState == "checking") { itemLbl.Text = "Checking " + itemName; itemR.Text = ""; indet = true; }
                    else if (itemTotal > 0)
                    {
                        itemLbl.Text = itemName;
                        itemR.Text = FormatSize(itemDone) + " of " + FormatSize(itemTotal);
                        indet = false;
                        p2 += (itemDone / itemTotal - p2) * 0.3;
                    }
                    else { itemLbl.Text = itemName; itemR.Text = ""; indet = true; }
                    detail.Text = itemsCount > 0 ? string.Format("Item {0} of {1}", itemIdx, itemsCount) : "";
                }
                else
                {
                    itemLbl.Text = ""; itemR.Text = ""; indet = true;
                    detail.Text = detailText;
                }
                marquee = (marquee + 0.012) % 1.0;
                bar2.Invalidate();
            };

            btn1.Click += (s, e) =>
            {
                string cs;
                lock (st.Sync) { cs = st.State; }
                if (cs == "running")
                {
                    DialogResult r = MessageBox.Show(f, "Stop setup now?\n\nNothing is lost: run setup again later and it continues where it stopped.",
                        "Clover AI Studio Setup", MessageBoxButtons.YesNo, MessageBoxIcon.Question);
                    if (r == DialogResult.Yes)
                    {
                        lock (st.Sync)
                        {
                            st.Cancel = true;
                            try { if (st.ActiveProcess != null && !st.ActiveProcess.HasExited) st.ActiveProcess.Kill(); } catch { }
                        }
                        btn1.Enabled = false;
                        btn1.Text = "Stopping";
                    }
                }
                else
                {
                    if (cs == "done") lock (st.Sync) { st.StartStudio = true; }
                    f.Close();
                }
            };

            btn2.Click += (s, e) =>
            {
                string cs;
                lock (st.Sync) { cs = st.State; }
                if (cs == "done") f.Close();
                else try { Process.Start("notepad.exe", "\"" + LogPath + "\""); } catch { }
            };

            f.FormClosing += (s, e) =>
            {
                string cs;
                lock (st.Sync) { cs = st.State; }
                if (cs == "running")
                {
                    e.Cancel = true;
                    if (btn1.Enabled) btn1.PerformClick();
                }
            };

            f.Shown += (s, e) => timer.Start();
            Application.Run(f);
        }

        private static string FormatSize(double bytes)
        {
            if (bytes >= 1024.0 * 1024.0 * 1024.0) return string.Format("{0:N1} GB", bytes / (1024.0 * 1024.0 * 1024.0));
            return string.Format("{0:N0} MB", bytes / (1024.0 * 1024.0));
        }
    }
}
