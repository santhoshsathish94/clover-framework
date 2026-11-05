using System;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Windows.Forms;
using Microsoft.Win32;

[assembly: AssemblyTitle("Clover AI Studio")]
[assembly: AssemblyDescription("Clover AI Studio local engine launcher and desktop window")]
[assembly: AssemblyCompany("Santhosh Narayanan")]
[assembly: AssemblyProduct("Clover AI Studio")]
[assembly: AssemblyCopyright("Copyright (c) 2026 Santhosh Narayanan")]
[assembly: AssemblyVersion("0.5.0.0")]
[assembly: AssemblyFileVersion("0.5.0.0")]
[assembly: ComVisible(false)]

namespace CloverAIStudio
{
    internal static class Program
    {
        private const int Port = 8765;
        private const int IdleMinutes = 30;
        private const int VsCodeGraceSeconds = 900;
        private const string AppTitle = "Clover AI Studio";

        [DllImport("user32.dll")]
        private static extern bool SetProcessDPIAware();

        [STAThread]
        private static int Main(string[] args)
        {
            try { SetProcessDPIAware(); } catch { }
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);

            string exePath = Assembly.GetExecutingAssembly().Location;
            string pkgDir = Path.GetDirectoryName(exePath) ?? AppDomain.CurrentDomain.BaseDirectory;
            string baseDir = pkgDir;

            bool engineOnly = false;
            bool stopOnly = false;
            bool uninstall = false;
            bool quiet = false;

            if (args != null)
            {
                foreach (string a in args)
                {
                    if (string.Equals(a, "--engine-only", StringComparison.OrdinalIgnoreCase)) engineOnly = true;
                    else if (string.Equals(a, "--stop", StringComparison.OrdinalIgnoreCase)) stopOnly = true;
                    else if (string.Equals(a, "--uninstall", StringComparison.OrdinalIgnoreCase)) uninstall = true;
                    else if (string.Equals(a, "--quiet", StringComparison.OrdinalIgnoreCase) ||
                             string.Equals(a, "-Quiet", StringComparison.OrdinalIgnoreCase)) quiet = true;
                }
            }

            if (uninstall)
            {
                return RunUninstall(baseDir, quiet);
            }

            if (stopOnly)
            {
                int code = HttpCall("POST", "/api/shutdown", "{}", 3000);
                if (!quiet)
                {
                    MessageBox.Show(
                        code == 200 ? "The Clover engine was stopped." : "The Clover engine was not running.",
                        AppTitle,
                        MessageBoxButtons.OK,
                        MessageBoxIcon.Information
                    );
                }
                return 0;
            }

            // When launched from an MSIX WindowsApps package (where pkgDir is read-only and has no venv),
            // use %USERPROFILE%\CloverAI as the writable runtime directory, syncing clover\ or running Setup on first start.
            if (!File.Exists(Path.Combine(baseDir, @"venv\Scripts\pythonw.exe")))
            {
                string userBase = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "CloverAI");
                string pkgClover = Path.Combine(pkgDir, "clover");
                if (Directory.Exists(pkgClover) && Directory.Exists(userBase))
                {
                    try { CopyDirectory(pkgClover, Path.Combine(userBase, "clover")); } catch { }
                }
                if (File.Exists(Path.Combine(userBase, @"venv\Scripts\pythonw.exe")) &&
                    File.Exists(Path.Combine(userBase, @"clover\server.py")))
                {
                    baseDir = userBase;
                }
                else
                {
                    string setupExe = Path.Combine(pkgDir, "CloverAIStudio-Setup.exe");
                    if (File.Exists(setupExe))
                    {
                        Process.Start(new ProcessStartInfo(setupExe, "--no-shortcuts") { UseShellExecute = true });
                        return 0;
                    }
                }
            }

            bool startedHere = false;
            if (!IsEngineUp())
            {
                Process proc = StartEngine(baseDir);
                if (proc == null) return 1;
                if (!WaitEngine(proc)) return 1;
                startedHere = true;
            }

            if (engineOnly) return 0;

            string url = "http://127.0.0.1:" + Port + "/";
            string edge = FindEdge();
            if (string.IsNullOrEmpty(edge))
            {
                Process.Start(new ProcessStartInfo(url) { UseShellExecute = true });
                return 0;
            }

            string profileDir = Path.Combine(baseDir, "studio-profile");
            string edgeArgs = string.Format(
                "--app={0} --user-data-dir=\"{1}\" --no-first-run --no-default-browser-check --disable-sync --disable-background-networking --disable-component-update --window-size=1240,880",
                url, profileDir
            );

            try
            {
                Stopwatch sw = Stopwatch.StartNew();
                ProcessStartInfo psi = new ProcessStartInfo(edge, edgeArgs)
                {
                    WorkingDirectory = baseDir,
                    UseShellExecute = false
                };
                using (Process edgeProc = Process.Start(psi))
                {
                    if (edgeProc != null)
                    {
                        edgeProc.WaitForExit();
                    }
                }
                if (startedHere && sw.ElapsedMilliseconds > 5000)
                {
                    HttpCall("POST", "/api/shutdown", "{\"unless_vscode_within_s\": " + VsCodeGraceSeconds + "}", 3000);
                }
            }
            catch
            {
                Process.Start(new ProcessStartInfo(url) { UseShellExecute = true });
            }

            return 0;
        }

        private static int HttpCall(string method, string path, string body, int timeoutMs)
        {
            try
            {
                HttpWebRequest req = (HttpWebRequest)WebRequest.Create("http://127.0.0.1:" + Port + path);
                req.Method = method;
                req.UserAgent = "CloverAIStudio";
                req.Timeout = timeoutMs;
                req.ReadWriteTimeout = timeoutMs;
                req.KeepAlive = false;
                if (body != null)
                {
                    byte[] data = Encoding.UTF8.GetBytes(body);
                    req.ContentType = "application/json";
                    req.ContentLength = data.Length;
                    using (Stream s = req.GetRequestStream())
                    {
                        s.Write(data, 0, data.Length);
                    }
                }
                using (HttpWebResponse resp = (HttpWebResponse)req.GetResponse())
                {
                    return (int)resp.StatusCode;
                }
            }
            catch (WebException wex)
            {
                HttpWebResponse r = wex.Response as HttpWebResponse;
                if (r != null) return (int)r.StatusCode;
                return -1;
            }
            catch
            {
                return -1;
            }
        }

        private static bool IsEngineUp()
        {
            return HttpCall("GET", "/api/status?passive=1", null, 2000) == 200;
        }

        private static void CopyDirectory(string sourceDir, string targetDir)
        {
            Directory.CreateDirectory(targetDir);
            foreach (string file in Directory.GetFiles(sourceDir))
            {
                string dest = Path.Combine(targetDir, Path.GetFileName(file));
                File.Copy(file, dest, true);
            }
            foreach (string sub in Directory.GetDirectories(sourceDir))
            {
                string name = Path.GetFileName(sub);
                if (string.Equals(name, "__pycache__", StringComparison.OrdinalIgnoreCase)) continue;
                CopyDirectory(sub, Path.Combine(targetDir, name));
            }
        }

        private static Process StartEngine(string baseDir)
        {
            string py = Path.Combine(baseDir, @"venv\Scripts\pythonw.exe");
            string script = Path.Combine(baseDir, @"clover\server.py");
            if (!File.Exists(py) || !File.Exists(script))
            {
                MessageBox.Show(
                    "The Clover runtime was not found in\n" + baseDir + "\n\nExpected venv\\Scripts\\pythonw.exe and clover\\server.py.",
                    AppTitle,
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );
                return null;
            }

            try
            {
                ProcessStartInfo psi = new ProcessStartInfo(py, string.Format("\"{0}\" --base \"{1}\" --idle-exit-min {2}", script, baseDir, IdleMinutes))
                {
                    WorkingDirectory = baseDir,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };
                psi.EnvironmentVariables["HF_HUB_OFFLINE"] = "1";
                psi.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";
                return Process.Start(psi);
            }
            catch (Exception ex)
            {
                MessageBox.Show("Could not start the Clover engine: " + ex.Message, AppTitle, MessageBoxButtons.OK, MessageBoxIcon.Error);
                return null;
            }
        }

        private static bool WaitEngine(Process proc)
        {
            for (int i = 0; i < 240; i++)
            {
                if (IsEngineUp()) return true;
                if (proc != null && proc.HasExited)
                {
                    int code = proc.ExitCode;
                    string msg = code == 3
                        ? "Port " + Port + " on 127.0.0.1 is used by another program, so the Clover engine cannot start."
                        : "The Clover engine stopped while starting (exit code " + code + "). See logs\\studio-server.log.";
                    MessageBox.Show(msg, AppTitle, MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return false;
                }
                Thread.Sleep(250);
            }
            MessageBox.Show("The Clover engine did not start within 60 seconds. See logs\\studio-server.log.", AppTitle, MessageBoxButtons.OK, MessageBoxIcon.Error);
            return false;
        }

        private static string FindEdge()
        {
            string[] envs = { "ProgramFiles(x86)", "ProgramFiles", "LOCALAPPDATA" };
            foreach (string v in envs)
            {
                string dir = Environment.GetEnvironmentVariable(v);
                if (!string.IsNullOrEmpty(dir))
                {
                    string candidate = Path.Combine(dir, @"Microsoft\Edge\Application\msedge.exe");
                    if (File.Exists(candidate)) return candidate;
                }
            }
            try
            {
                using (RegistryKey k = Registry.LocalMachine.OpenSubKey(@"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\msedge.exe"))
                {
                    if (k != null)
                    {
                        string val = k.GetValue(null) as string;
                        if (!string.IsNullOrEmpty(val) && File.Exists(val)) return val;
                    }
                }
            }
            catch { }
            return null;
        }

        private static int RunUninstall(string baseDir, bool quiet)
        {
            HttpCall("POST", "/api/shutdown", "{}", 3000);
            Thread.Sleep(1500);

            string programs = Environment.GetFolderPath(Environment.SpecialFolder.Programs);
            string desktop = Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory);
            string[] links = {
                Path.Combine(programs, "Clover AI Studio.lnk"),
                Path.Combine(programs, "Clover Engine for VS Code.lnk"),
                Path.Combine(desktop, "Clover AI Studio.lnk")
            };
            foreach (string l in links)
            {
                try { if (File.Exists(l)) File.Delete(l); } catch { }
            }

            string[] codePaths = {
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), @"Programs\Microsoft VS Code\bin\code.cmd"),
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles), @"Microsoft VS Code\bin\code.cmd")
            };
            foreach (string cp in codePaths)
            {
                if (File.Exists(cp))
                {
                    try
                    {
                        ProcessStartInfo psi = new ProcessStartInfo(cp, "--uninstall-extension clover-framework.clover-local")
                        {
                            UseShellExecute = false,
                            CreateNoWindow = true
                        };
                        using (Process p = Process.Start(psi))
                        {
                            if (p != null) p.WaitForExit(15000);
                        }
                    }
                    catch { }
                    break;
                }
            }

            try
            {
                Registry.CurrentUser.DeleteSubKeyTree(@"Software\Microsoft\Windows\CurrentVersion\Uninstall\CloverAIStudio", false);
            }
            catch { }

            string profileDir = Path.Combine(baseDir, "studio-profile");
            try { if (Directory.Exists(profileDir)) Directory.Delete(profileDir, true); } catch { }

            string msg = "Clover AI Studio shortcuts and VS Code extension were removed.\n\nThe folder " + baseDir +
                         " was kept: it holds Python, the models and tools, and your own data (chats, memory, and what Clover learned about you). Delete that folder to remove everything.";
            if (quiet) Console.WriteLine(msg);
            else MessageBox.Show(msg, AppTitle, MessageBoxButtons.OK, MessageBoxIcon.Information);
            return 0;
        }
    }
}
