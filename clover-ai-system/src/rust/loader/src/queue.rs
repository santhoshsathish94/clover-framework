//! The per-card queues and the pending counter. Behaviour reference: `struct Loader`, `g_loader[]`, `pending_mu()/pending_cv()/g_pending`, `loader_submit`, `loader_wait`
//! (backend-gpu.cu 538-572 at 4d8d891). Jobs are plain closures here; the thread-start hook (the original's `CHECK(cudaSetDevice(card))`) is a parameter so the queue can be tested
//! without any callback table.
use crate::sys::fatal_msg;
use std::collections::VecDeque;
use std::sync::{Condvar, Mutex, MutexGuard};

pub const MAX_CARDS: usize = 8; // backend-gpu.cu:31

pub type Job = Box<dyn FnOnce() + Send + 'static>;

struct CardState {
    started: bool,
    q: VecDeque<Job>,
}
struct CardQ {
    st: Mutex<CardState>,
    cv: Condvar,
}

pub struct Loader {
    cards: [CardQ; MAX_CARDS],
    pending: Mutex<usize>,
    pending_cv: Condvar,
    on_start: Box<dyn Fn(usize) + Send + Sync>,
}

// A panic can only come from a job or the hook, and neither runs while a queue lock is held, so a poisoned lock never hides half-updated data.
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl Loader {
    /// Leaked on purpose: the worker threads are never stopped, so the loader lives for the process (`g_loader[]` is never destroyed in the original).
    pub fn new(on_start: Box<dyn Fn(usize) + Send + Sync>) -> &'static Loader {
        Box::leak(Box::new(Loader {
            cards: std::array::from_fn(|_| CardQ { st: Mutex::new(CardState { started: false, q: VecDeque::new() }), cv: Condvar::new() }),
            pending: Mutex::new(0),
            pending_cv: Condvar::new(),
            on_start,
        }))
    }

    /// loader_submit. `card` must be below MAX_CARDS (the C entry points check it and print the message).
    pub fn submit(&'static self, card: usize, job: Job) {
        // pending is raised before the job is queued and before the card's lock is taken, as in the original
        *lock(&self.pending) += 1;
        let c = &self.cards[card];
        let mut g = lock(&c.st);
        if !g.started {
            g.started = true;
            let me: &'static Loader = self;
            // the handle is dropped: detached, never joined; std::thread's default in the original is the system's 8 MB stack
            if let Err(e) = std::thread::Builder::new().stack_size(8 << 20).spawn(move || me.worker(card)) {
                fatal_msg(format!("loader: cannot start the thread of card {}: {}\n", card, e).as_bytes());
            }
        }
        g.q.push_back(job);
        c.cv.notify_one();
    }

    /// loader_wait.
    pub fn wait(&self) {
        let mut p = lock(&self.pending);
        while *p != 0 {
            p = self.pending_cv.wait(p).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Jobs submitted and not finished (for tests).
    pub fn pending(&self) -> usize {
        *lock(&self.pending)
    }

    fn worker(&'static self, card: usize) {
        (self.on_start)(card);
        let c = &self.cards[card];
        loop {
            let job = {
                let mut g = lock(&c.st);
                loop {
                    if let Some(j) = g.q.pop_front() {
                        break j;
                    }
                    g = c.cv.wait(g).unwrap_or_else(|e| e.into_inner());
                }
            };
            job();
            *lock(&self.pending) -= 1;
            self.pending_cv.notify_all();
        }
    }
}
