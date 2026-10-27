//! Definition of the [`SemaphoreList`] type.

use alloc::vec::Vec;
use ash::vk;
use core::mem::MaybeUninit;

#[derive(Debug, PartialEq)]
pub enum SemaphoreListMode {
    Wait,
    Signal,
}

/// A list of Vulkan semaphores to wait for or signal.
///
/// This represents a list of binary or timeline semaphores, together
/// with values for the timeline semaphores, and stage masks, if these
/// are used for waiting.
///
/// This type ensures that the array of semaphores to be signaled
/// stays aligned with the array of values for timeline semaphores
/// appearing in that list. The [`add_to_submit`] method prepares the
/// `vkQueueSubmit` arguments appropriately for whatever semaphores we
/// actually have.
///
/// [`add_to_submit`]: SemaphoreList::add_to_submit
#[derive(Debug)]
pub struct SemaphoreList {
    /// Mode of the semaphore list. Used for validation.
    mode: SemaphoreListMode,

    /// Semaphores to use.
    ///
    /// This can be a mix of binary and timeline semaphores.
    semaphores: Vec<vk::Semaphore>,

    /// Values for the timeline semaphores.
    ///
    /// This may be empty for binary-only lists until [`Self::add_to_submit`].
    /// When populated, this has the same length as [`semaphores`], with
    /// dummy !0 values for binary semaphores, since Vulkan ignores these.
    ///
    /// [`semaphores`]: Self::semaphores
    values: Vec<u64>,

    /// Stage masks for wait semaphores.
    ///
    /// This is only used if `mode` is `Wait`.
    pub stage_masks: Vec<vk::PipelineStageFlags>,
}

impl SemaphoreList {
    pub fn new(mode: SemaphoreListMode) -> Self {
        Self {
            mode,
            semaphores: Vec::new(),
            values: Vec::new(),
            stage_masks: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.semaphores.is_empty()
    }

    /// Add this list to the semaphores to be signalled by a `vkQueueSubmit` call.
    ///
    /// - Set `submit_info`'s `pSignalSemaphores` list to this list's
    ///   semaphores.
    ///
    /// - If this list contains any timeline semaphores, then initialize
    ///   `timeline_info`, set its `pSignalSemaphoreValues` to this
    ///   list's values, and add it to `submit_info`s extension chain.
    ///
    /// Return the revised `submit_info` value.
    pub fn add_to_submit<'info, 'semaphores: 'info>(
        wait_semaphores: &'semaphores mut Self,
        signal_semaphores: &'semaphores mut Self,
        submit_info: vk::SubmitInfo<'info>,
        timeline_info: &'info mut MaybeUninit<vk::TimelineSemaphoreSubmitInfo<'info>>,
    ) -> vk::SubmitInfo<'info> {
        wait_semaphores.check();
        signal_semaphores.check();

        assert!(matches!(wait_semaphores.mode, SemaphoreListMode::Wait));
        assert!(matches!(signal_semaphores.mode, SemaphoreListMode::Signal));

        let timeline_info = timeline_info.write(vk::TimelineSemaphoreSubmitInfo::default());

        // Some drivers read binary semaphore values whenever timeline info is chained.
        // https://github.com/gfx-rs/wgpu/pull/10677
        if !wait_semaphores.values.is_empty() || !signal_semaphores.values.is_empty() {
            wait_semaphores.pad_values();
            signal_semaphores.pad_values();
        }

        let mut uses_timeline = false;

        if !wait_semaphores.values.is_empty() {
            *timeline_info = timeline_info.wait_semaphore_values(&wait_semaphores.values);
            uses_timeline = true;
        }

        if !signal_semaphores.values.is_empty() {
            *timeline_info = timeline_info.signal_semaphore_values(&signal_semaphores.values);
            uses_timeline = true;
        }

        let mut submit_info = submit_info
            .wait_semaphores(&wait_semaphores.semaphores)
            .wait_dst_stage_mask(&wait_semaphores.stage_masks)
            .signal_semaphores(&signal_semaphores.semaphores);

        if uses_timeline {
            submit_info = submit_info.push_next(timeline_info);
        }

        submit_info
    }

    /// Add a semaphore to be signaled. Panics if this is a list of semaphores to wait.
    pub fn push_signal(&mut self, semaphore: SemaphoreType) {
        assert!(matches!(self.mode, SemaphoreListMode::Signal));
        self.push_inner(semaphore);
    }

    /// Add a semaphore to be waited for. Panics if this is a list of semaphores to signal.
    pub fn push_wait(&mut self, semaphore: SemaphoreType, stage: vk::PipelineStageFlags) {
        assert!(matches!(self.mode, SemaphoreListMode::Wait));

        self.stage_masks.push(stage);
        self.push_inner(semaphore);
    }

    fn push_inner(&mut self, semaphore: SemaphoreType) {
        match semaphore {
            SemaphoreType::Binary(semaphore) => {
                self.semaphores.push(semaphore);
                // Push a dummy value if necessary.
                if !self.values.is_empty() {
                    self.values.push(!0);
                }
            }
            SemaphoreType::Timeline(semaphore, value) => {
                // We may be the first timeline semaphore, ensure that the values
                // array is filled with dummy values for existing binary semaphores.
                self.pad_values();
                self.semaphores.push(semaphore);
                self.values.push(value);
            }
        }

        self.check();
    }

    /// Remove all entries matching `semaphore` from the list. Returns `true`
    /// if anything was removed.
    pub fn remove(&mut self, semaphore: vk::Semaphore) -> bool {
        let mut removed = false;
        let mut i = 0;
        while i < self.semaphores.len() {
            if self.semaphores[i] == semaphore {
                self.semaphores.swap_remove(i);
                if !self.values.is_empty() {
                    self.values.swap_remove(i);
                }
                if !self.stage_masks.is_empty() {
                    self.stage_masks.swap_remove(i);
                }
                removed = true;
            } else {
                i += 1;
            }
        }
        self.check();
        removed
    }

    /// Append `other` to `self`, leaving `other` empty.
    pub fn append(&mut self, other: &mut Self) {
        assert_eq!(self.mode, other.mode);

        // If we're about to receive values, ensure we're aligned first.
        if !other.values.is_empty() {
            self.pad_values();
        }
        self.semaphores.append(&mut other.semaphores);
        self.values.append(&mut other.values);
        // If we had values, but `other` did not, re-align.
        if !self.values.is_empty() {
            self.pad_values();
        }
        self.stage_masks.append(&mut other.stage_masks);
        self.check();
    }

    /// Pad `self.values` with dummy values for binary semaphores,
    /// in preparation for adding a timeline semaphore value.
    ///
    /// This is a no-op if we already have values.
    fn pad_values(&mut self) {
        self.values.resize(self.semaphores.len(), !0);
    }

    #[track_caller]
    fn check(&self) {
        debug_assert!(self.values.is_empty() || self.values.len() == self.semaphores.len());
        match self.mode {
            SemaphoreListMode::Wait => {
                debug_assert!(
                    self.stage_masks.is_empty() || self.stage_masks.len() == self.semaphores.len()
                );
            }
            SemaphoreListMode::Signal => {
                debug_assert!(self.stage_masks.is_empty());
            }
        }
    }
}

pub enum SemaphoreType {
    Binary(vk::Semaphore),
    Timeline(vk::Semaphore, u64),
}

#[cfg(test)]
mod tests {
    use ash::vk::Handle;

    use super::*;

    fn assert_submit_values(
        mut wait: SemaphoreList,
        mut signal: SemaphoreList,
        expected_wait: &[u64],
        expected_signal: &[u64],
        uses_timeline: bool,
    ) {
        let wait_count = wait.semaphores.len() as u32;
        let signal_count = signal.semaphores.len() as u32;
        let mut timeline = MaybeUninit::uninit();
        let submit = SemaphoreList::add_to_submit(
            &mut wait,
            &mut signal,
            vk::SubmitInfo::default(),
            &mut timeline,
        );

        assert_eq!(submit.wait_semaphore_count, wait_count);
        assert_eq!(submit.signal_semaphore_count, signal_count);
        let (wait_values, signal_values) = if uses_timeline {
            assert!(!submit.p_next.is_null());
            // SAFETY: add_to_submit chains initialized timeline storage that is still alive.
            let timeline = unsafe { &*submit.p_next.cast::<vk::TimelineSemaphoreSubmitInfo<'_>>() };
            assert_eq!(
                timeline.wait_semaphore_value_count as usize,
                expected_wait.len()
            );
            assert_eq!(
                timeline.signal_semaphore_value_count as usize,
                expected_signal.len()
            );
            (
                timeline.p_wait_semaphore_values,
                timeline.p_signal_semaphore_values,
            )
        } else {
            assert!(submit.p_next.is_null());
            (core::ptr::null(), core::ptr::null())
        };
        assert_eq!(wait.values, expected_wait);
        assert_eq!(signal.values, expected_signal);
        assert_eq!(
            wait_values,
            if expected_wait.is_empty() {
                core::ptr::null()
            } else {
                wait.values.as_ptr()
            }
        );
        assert_eq!(
            signal_values,
            if expected_signal.is_empty() {
                core::ptr::null()
            } else {
                signal.values.as_ptr()
            }
        );
    }

    #[test]
    fn binary_wait_with_timeline_signal_gets_dummy_value() {
        let mut wait = SemaphoreList::new(SemaphoreListMode::Wait);
        wait.push_wait(
            SemaphoreType::Binary(vk::Semaphore::from_raw(1)),
            vk::PipelineStageFlags::ALL_COMMANDS,
        );
        let mut signal = SemaphoreList::new(SemaphoreListMode::Signal);
        signal.push_signal(SemaphoreType::Timeline(vk::Semaphore::from_raw(2), 7));

        assert_submit_values(wait, signal, &[!0], &[7], true);
    }

    #[test]
    fn binary_signal_with_timeline_wait_gets_dummy_value() {
        let mut wait = SemaphoreList::new(SemaphoreListMode::Wait);
        wait.push_wait(
            SemaphoreType::Timeline(vk::Semaphore::from_raw(1), 7),
            vk::PipelineStageFlags::ALL_COMMANDS,
        );
        let mut signal = SemaphoreList::new(SemaphoreListMode::Signal);
        signal.push_signal(SemaphoreType::Binary(vk::Semaphore::from_raw(2)));

        assert_submit_values(wait, signal, &[7], &[!0], true);
    }

    #[test]
    fn mixed_semaphores_preserve_timeline_values() {
        let mut wait = SemaphoreList::new(SemaphoreListMode::Wait);
        wait.push_wait(
            SemaphoreType::Binary(vk::Semaphore::from_raw(1)),
            vk::PipelineStageFlags::ALL_COMMANDS,
        );
        wait.push_wait(
            SemaphoreType::Timeline(vk::Semaphore::from_raw(2), 7),
            vk::PipelineStageFlags::ALL_COMMANDS,
        );
        let mut signal = SemaphoreList::new(SemaphoreListMode::Signal);
        signal.push_signal(SemaphoreType::Timeline(vk::Semaphore::from_raw(3), 9));
        signal.push_signal(SemaphoreType::Binary(vk::Semaphore::from_raw(4)));

        assert_submit_values(wait, signal, &[!0, 7], &[9, !0], true);
    }

    #[test]
    fn binary_only_submission_has_no_timeline_info() {
        let mut wait = SemaphoreList::new(SemaphoreListMode::Wait);
        wait.push_wait(
            SemaphoreType::Binary(vk::Semaphore::from_raw(1)),
            vk::PipelineStageFlags::ALL_COMMANDS,
        );
        let mut signal = SemaphoreList::new(SemaphoreListMode::Signal);
        signal.push_signal(SemaphoreType::Binary(vk::Semaphore::from_raw(2)));

        assert_submit_values(wait, signal, &[], &[], false);
    }

    #[test]
    fn timeline_signal_with_no_waits_keeps_wait_values_empty() {
        let wait = SemaphoreList::new(SemaphoreListMode::Wait);
        let mut signal = SemaphoreList::new(SemaphoreListMode::Signal);
        signal.push_signal(SemaphoreType::Timeline(vk::Semaphore::from_raw(1), 7));

        assert_submit_values(wait, signal, &[], &[7], true);
    }

    #[test]
    fn empty_submission_has_no_timeline_info() {
        assert_submit_values(
            SemaphoreList::new(SemaphoreListMode::Wait),
            SemaphoreList::new(SemaphoreListMode::Signal),
            &[],
            &[],
            false,
        );
    }
}
