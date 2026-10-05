'use strict';

function plainArgs(value, keys) {
  if (!value || typeof value !== 'object' || Array.isArray(value) || ![Object.prototype, null].includes(Object.getPrototypeOf(value))) throw new Error('Arguments must be an object');
  if (Object.keys(value).some(key => !keys.includes(key))) throw new Error('Unsupported operation argument');
  if (Buffer.byteLength(JSON.stringify(value)) > 512 * 1024) throw new Error('Arguments exceed the local limit');
  return value;
}

function assertSender(event, window, allowedUrl) {
  if (!window || window.isDestroyed() || event.sender !== window.webContents
      || event.senderFrame !== window.webContents.mainFrame
      || event.senderFrame.url !== allowedUrl) throw new Error('Untrusted desktop frame');
}

module.exports = { plainArgs, assertSender };
