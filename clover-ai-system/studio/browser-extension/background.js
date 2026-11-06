// Send to Clover: when the user clicks the button or a menu item, and only then, the current tab is read (activeTab)
// and handed to Clover AI Studio on this computer through native messaging. The browser starts Clover's host program
// for this extension's ID only; nothing is sent anywhere else and nothing runs in the background.
const HOST = 'com.cloverframework.studio';
const LIMITS = { html: 3000000, text: 1000000, selection: 100000 };

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({ id: 'send-page', title: 'Send page to Clover', contexts: ['page'] });
  chrome.contextMenus.create({ id: 'send-selection', title: 'Send page and selection to Clover', contexts: ['selection'] });
});

function grab(limits) {  // runs in the page, after the user's click
  const selection = String(window.getSelection ? window.getSelection() : '');
  return {
    url: location.href,
    title: document.title,
    selection: selection.slice(0, limits.selection),
    html: document.documentElement.outerHTML.slice(0, limits.html),
    text: (document.body ? document.body.innerText : '').slice(0, limits.text)
  };
}

function show(tab, ok, title) {
  const tabId = tab && tab.id;
  chrome.action.setBadgeText({ tabId, text: ok ? '\u2713' : '!' });
  chrome.action.setBadgeBackgroundColor({ tabId, color: ok ? '#1e7a3c' : '#b3261e' });
  chrome.action.setTitle({ tabId, title });
}

async function send(tab) {
  if (!tab || !/^https?:/i.test(tab.url || '')) {
    show(tab, false, 'Only web pages (http or https) can be sent to Clover.');
    return;
  }
  try {
    const [res] = await chrome.scripting.executeScript({ target: { tabId: tab.id }, func: grab, args: [LIMITS] });
    const reply = await chrome.runtime.sendNativeMessage(HOST, Object.assign({ type: 'page' }, res.result));
    if (reply && reply.ok) show(tab, true, 'Sent to Clover. In Clover AI Studio, add it to your message.');
    else show(tab, false, 'Clover did not take the page: ' + ((reply && reply.error) || 'no answer'));
  } catch (e) {
    const msg = String((e && e.message) || e);
    show(tab, false, /host not found/i.test(msg)
      ? 'Clover AI Studio is not set up for this browser yet: in Clover, Settings > Tools and search > Browser > Set up.'
      : 'Could not send the page: ' + msg);
  }
}

chrome.action.onClicked.addListener(send);
chrome.contextMenus.onClicked.addListener((info, tab) => send(tab));
