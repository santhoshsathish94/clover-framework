// Protected verification artifact for the runtime-enforcement example.
const example = require('../src/app');

if (example() !== 'implementation can be changed') {
  throw new Error('verification failed');
}
