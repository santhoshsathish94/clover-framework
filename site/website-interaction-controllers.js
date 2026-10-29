/* Progressive enhancement only: every panel is in the DOM and readable without JS.
 *
 * Object-Oriented Architecture & Complete-English Role-Prefix Naming (`site/app.js`):
 * All website behaviors are organized into 4 cohesive domain classes using explicit role prefixes
 * (`class_*`, `function_*`, `variable_*`, `flag_*`, `constant_*`) so every identifier's role and
 * purpose is immediately clear:
 *   1. `class_viewport_scroll_navigator` (`ViewportScrollNavigator` -> `ModalityCodec` / `Stage2_Direction`)
 *   2. `class_five_stage_clover_mark_controller` (`FiveStageCloverMarkController` -> `CloverCycleCoordinator`)
 *   3. `class_site_navigation_drawer_controller` (`SiteNavigationDrawerController` -> `TaskDirection`)
 *   4. `class_accessible_tab_and_glossary_controller` (`AccessibleTabAndGlossaryController` -> `VerifiedKnowledgeGraph`)
 */
(function () {
  'use strict';

  // ==========================================================================
  // CONSTANTS — `constant_*` (Complete Quantified Noun Phrases)
  // ==========================================================================

  var constant_minimum_scroll_duration_milliseconds = 450;
  var constant_maximum_scroll_duration_milliseconds = 1200;
  var constant_scroll_distance_duration_multiplier = 0.6;
  var constant_full_clover_rotation_degrees = 360;
  var constant_clover_rotation_easing_factor = 0.12;
  var constant_closing_section_viewport_ratio = 0.9;
  var constant_desktop_navigation_breakpoint_pixels = 999;
  var constant_real_world_muted_clover_aria_label =
    'The same five-leaf cycle in muted grey, unchanged in shape, drained of its colour.';

  // ==========================================================================
  // 1. `class_viewport_scroll_navigator` — Eased Anchor Scrolling & Hash Alignment
  // ==========================================================================

  class class_viewport_scroll_navigator {
    constructor() {
      var variable_reduced_motion_media_query =
        window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)');
      this.flag_prefers_reduced_motion = !!(
        variable_reduced_motion_media_query && variable_reduced_motion_media_query.matches
      );
      this.flag_can_animate_scroll =
        !this.flag_prefers_reduced_motion && typeof window.requestAnimationFrame === 'function';

      if (this.flag_can_animate_scroll) {
        document.documentElement.style.scrollBehavior = 'auto';
      }
    }

    function_ease_in_out_cubic_progress(variable_normalized_time) {
      return variable_normalized_time < 0.5
        ? 4 * variable_normalized_time * variable_normalized_time * variable_normalized_time
        : 1 - Math.pow(-2 * variable_normalized_time + 2, 3) / 2;
    }

    function_calculate_target_scroll_offset(variable_target_element, variable_alignment_mode) {
      var variable_bounding_rectangle = variable_target_element.getBoundingClientRect();
      var variable_absolute_top_pixels = variable_bounding_rectangle.top + window.scrollY;
      if (variable_alignment_mode === 'center') {
        return variable_absolute_top_pixels - (window.innerHeight - variable_bounding_rectangle.height) / 2;
      }
      var variable_scroll_margin_top_pixels =
        parseFloat(getComputedStyle(variable_target_element).scrollMarginTop) || 0;
      return variable_absolute_top_pixels - variable_scroll_margin_top_pixels;
    }

    function_scroll_viewport_to_vertical_offset(variable_destination_y_pixels) {
      if (!this.flag_can_animate_scroll) {
        window.scrollTo(0, variable_destination_y_pixels);
        return;
      }

      var variable_starting_y_pixels = window.scrollY;
      var variable_travel_distance_pixels = variable_destination_y_pixels - variable_starting_y_pixels;
      if (Math.abs(variable_travel_distance_pixels) < 2) {
        return;
      }

      var variable_animation_duration_milliseconds = Math.min(
        constant_maximum_scroll_duration_milliseconds,
        Math.max(
          constant_minimum_scroll_duration_milliseconds,
          Math.abs(variable_travel_distance_pixels) * constant_scroll_distance_duration_multiplier
        )
      );
      var variable_animation_start_timestamp = null;
      var variable_self_navigator = this;

      (function function_step_scroll_animation_frame(variable_current_timestamp) {
        if (variable_animation_start_timestamp === null) {
          variable_animation_start_timestamp = variable_current_timestamp;
        }
        var variable_elapsed_progress = Math.min(
          1,
          (variable_current_timestamp - variable_animation_start_timestamp) /
            variable_animation_duration_milliseconds
        );
        var variable_eased_progress =
          variable_self_navigator.function_ease_in_out_cubic_progress(variable_elapsed_progress);
        window.scrollTo(
          0,
          variable_starting_y_pixels + variable_travel_distance_pixels * variable_eased_progress
        );
        if (variable_elapsed_progress < 1) {
          window.requestAnimationFrame(function_step_scroll_animation_frame);
        }
      })(performance.now());
    }

    function_scroll_viewport_to_element(variable_target_element, variable_alignment_mode) {
      var variable_maximum_scrollable_offset =
        document.documentElement.scrollHeight - window.innerHeight;
      var variable_clamped_destination_y = Math.max(
        0,
        Math.min(
          variable_maximum_scrollable_offset,
          this.function_calculate_target_scroll_offset(variable_target_element, variable_alignment_mode)
        )
      );
      this.function_scroll_viewport_to_vertical_offset(variable_clamped_destination_y);
    }

    function_check_if_link_points_to_current_page(variable_anchor_element) {
      if (
        variable_anchor_element.protocol !== window.location.protocol ||
        variable_anchor_element.host !== window.location.host
      ) {
        return false;
      }
      var function_strip_index_suffix = function (variable_pathname_string) {
        return variable_pathname_string.replace(/index\.html$/, '');
      };
      return (
        function_strip_index_suffix(variable_anchor_element.pathname) ===
        function_strip_index_suffix(window.location.pathname)
      );
    }

    function_align_viewport_to_initial_url_hash() {
      var variable_initial_hash = window.location.hash;
      if (!variable_initial_hash) {
        return;
      }
      var variable_target_element = document.getElementById(variable_initial_hash.slice(1));
      if (!variable_target_element) {
        return;
      }
      var variable_self_navigator = this;
      var function_perform_initial_alignment = function () {
        window.requestAnimationFrame(function () {
          if (window.location.hash !== variable_initial_hash) {
            return;
          }
          var variable_maximum_scrollable_offset =
            document.documentElement.scrollHeight - window.innerHeight;
          var flag_is_centered_stage_section =
            variable_target_element.classList.contains('stage') &&
            !variable_target_element.classList.contains('responsibility-band');
          var variable_alignment_mode = flag_is_centered_stage_section ? 'center' : 'start';
          window.scrollTo(
            0,
            Math.max(
              0,
              Math.min(
                variable_maximum_scrollable_offset,
                variable_self_navigator.function_calculate_target_scroll_offset(
                  variable_target_element,
                  variable_alignment_mode
                )
              )
            )
          );
        });
      };

      if (document.fonts && document.fonts.ready) {
        document.fonts.ready.then(function_perform_initial_alignment);
      } else {
        function_perform_initial_alignment();
      }
    }

    function_bind_document_anchor_click_delegation() {
      var variable_self_navigator = this;
      if (window.location.hash) {
        window.addEventListener(
          'load',
          function () {
            variable_self_navigator.function_align_viewport_to_initial_url_hash();
          },
          { once: true }
        );
      }

      document.addEventListener('click', function (variable_click_event) {
        var variable_clicked_anchor =
          variable_click_event.target.closest && variable_click_event.target.closest('a[href]');
        if (!variable_clicked_anchor) {
          return;
        }
        var flag_has_modifier_key =
          variable_clicked_anchor.target === '_blank' ||
          variable_click_event.metaKey ||
          variable_click_event.ctrlKey ||
          variable_click_event.shiftKey ||
          variable_click_event.altKey;
        if (flag_has_modifier_key) {
          return;
        }

        var variable_raw_href_attribute = variable_clicked_anchor.getAttribute('href') || '';

        if (
          variable_raw_href_attribute.charAt(0) !== '#' &&
          !variable_clicked_anchor.hash &&
          variable_self_navigator.function_check_if_link_points_to_current_page(variable_clicked_anchor)
        ) {
          variable_click_event.preventDefault();
          variable_self_navigator.function_scroll_viewport_to_vertical_offset(0);
          if (window.location.hash && history.pushState) {
            history.pushState(null, '', window.location.pathname + window.location.search);
          }
          return;
        }

        if (
          window.location.protocol === 'file:' &&
          variable_clicked_anchor.protocol === 'file:' &&
          /\/$/.test(variable_clicked_anchor.pathname)
        ) {
          variable_click_event.preventDefault();
          window.location.href = variable_clicked_anchor.href.replace(/\/(#.*)?$/, '/index.html$1');
          return;
        }

        if (variable_raw_href_attribute.charAt(0) !== '#' || variable_raw_href_attribute === '#') {
          return;
        }

        var variable_target_element = document.getElementById(variable_raw_href_attribute.slice(1));
        if (!variable_target_element) {
          return;
        }

        variable_click_event.preventDefault();
        var variable_alignment_mode = variable_target_element.classList.contains('stage')
          ? 'center'
          : 'start';
        variable_self_navigator.function_scroll_viewport_to_element(
          variable_target_element,
          variable_alignment_mode
        );
        if (history.pushState) {
          history.pushState(null, '', variable_raw_href_attribute);
        } else {
          window.location.hash = variable_raw_href_attribute;
        }
      });
    }
  }

  // ==========================================================================
  // 2. `class_five_stage_clover_mark_controller` — Pinned 5-Leaf Cycle Wheel & Real-World Transition
  // ==========================================================================

  class class_five_stage_clover_mark_controller {
    constructor(variable_prefers_reduced_motion) {
      this.flag_prefers_reduced_motion = !!variable_prefers_reduced_motion;
      this.variable_story_containers = Array.prototype.slice.call(
        document.querySelectorAll('[data-story]')
      );
      this.variable_spinning_wheel_element = document.querySelector('.pinned__mark .clover__spin');
      this.variable_pinned_mark_container = document.querySelector('.pinned__mark');
      this.variable_first_real_world_section = document.getElementById('evidence-preview');
      this.variable_closing_answer_section = document.getElementById('what-now');
    }

    function_initialize_stage_highlighting_and_wheel_rotation() {
      this.function_bind_five_stage_leaf_highlighting();
      this.function_bind_continuous_wheel_rotation();
      this.function_bind_real_world_palette_and_link_transition();
      this.function_bind_closing_section_visibility();
    }

    function_bind_five_stage_leaf_highlighting() {
      if (!this.variable_story_containers.length) {
        return;
      }
      var variable_story_descriptors = this.variable_story_containers.map(function (
        variable_story_element
      ) {
        var variable_stage_sections = Array.prototype.slice.call(
          variable_story_element.querySelectorAll('.stage')
        );
        var variable_leaf_and_label_marks = Array.prototype.slice.call(
          variable_story_element.querySelectorAll('.story__leaf, .story__label')
        );
        var variable_caption_element = variable_story_element.querySelector('[data-story-caption]');

        var function_apply_active_stage = function (variable_active_stage_section) {
          var variable_active_leaf_name = variable_active_stage_section
            ? variable_active_stage_section.getAttribute('data-leaf') || ''
            : '';
          variable_leaf_and_label_marks.forEach(function (variable_mark_element) {
            var flag_matches_active_leaf =
              variable_active_leaf_name !== '' &&
              variable_mark_element.getAttribute('data-leaf') === variable_active_leaf_name;
            variable_mark_element.classList.toggle('is-active', flag_matches_active_leaf);
          });
          variable_stage_sections.forEach(function (variable_section_element) {
            variable_section_element.classList.toggle(
              'is-current',
              variable_section_element === variable_active_stage_section
            );
          });
          var flag_should_ink_mark =
            !!variable_active_stage_section && variable_active_stage_section.hasAttribute('data-ink');
          variable_story_element.classList.toggle('is-ink', flag_should_ink_mark);
          if (
            variable_active_stage_section &&
            variable_caption_element &&
            variable_active_stage_section.getAttribute('data-caption')
          ) {
            variable_caption_element.textContent =
              variable_active_stage_section.getAttribute('data-caption');
          }
        };

        return {
          variable_stage_sections: variable_stage_sections,
          function_apply_active_stage: function_apply_active_stage
        };
      });

      var function_synchronize_active_stage_with_viewport = function () {
        var variable_viewport_midpoint_y = window.innerHeight / 2;
        variable_story_descriptors.forEach(function (variable_descriptor) {
          var variable_closest_stage_section = null;
          var variable_minimum_center_distance = Infinity;
          var flag_is_reader_inside_stage_band = false;

          variable_descriptor.variable_stage_sections.forEach(function (variable_section_element) {
            var variable_rect = variable_section_element.getBoundingClientRect();
            if (
              variable_rect.top <= variable_viewport_midpoint_y &&
              variable_rect.bottom >= variable_viewport_midpoint_y
            ) {
              flag_is_reader_inside_stage_band = true;
            }
            var variable_distance_to_midpoint = Math.abs(
              variable_rect.top + variable_rect.height / 2 - variable_viewport_midpoint_y
            );
            if (variable_distance_to_midpoint < variable_minimum_center_distance) {
              variable_minimum_center_distance = variable_distance_to_midpoint;
              variable_closest_stage_section = variable_section_element;
            }
          });

          variable_descriptor.function_apply_active_stage(
            flag_is_reader_inside_stage_band ? variable_closest_stage_section : null
          );
        });
      };

      var flag_is_scroll_frame_queued = false;
      var function_on_viewport_scroll_or_resize = function () {
        if (flag_is_scroll_frame_queued) {
          return;
        }
        flag_is_scroll_frame_queued = true;
        window.requestAnimationFrame(function () {
          flag_is_scroll_frame_queued = false;
          function_synchronize_active_stage_with_viewport();
        });
      };

      function_synchronize_active_stage_with_viewport();
      window.addEventListener('scroll', function_on_viewport_scroll_or_resize, { passive: true });
      window.addEventListener('resize', function_on_viewport_scroll_or_resize);
    }

    function_bind_continuous_wheel_rotation() {
      if (!this.variable_spinning_wheel_element || this.flag_prefers_reduced_motion) {
        return;
      }
      var variable_wheel_element = this.variable_spinning_wheel_element;
      var variable_current_rotation_degrees = 0;
      var variable_target_rotation_degrees = 0;
      var flag_is_wheel_animation_running = false;

      var function_read_target_rotation_from_scroll = function () {
        var variable_total_scrollable_span =
          document.documentElement.scrollHeight - window.innerHeight;
        var variable_scroll_fraction =
          variable_total_scrollable_span > 0
            ? Math.min(1, Math.max(0, window.pageYOffset / variable_total_scrollable_span))
            : 0;
        variable_target_rotation_degrees =
          variable_scroll_fraction * constant_full_clover_rotation_degrees;
      };

      var function_render_wheel_transform = function () {
        variable_wheel_element.setAttribute(
          'transform',
          'rotate(' + variable_current_rotation_degrees.toFixed(2) + ' 50 44)'
        );
      };

      var function_ease_wheel_rotation_step = function () {
        var variable_angular_gap = variable_target_rotation_degrees - variable_current_rotation_degrees;
        if (Math.abs(variable_angular_gap) < 0.05) {
          variable_current_rotation_degrees = variable_target_rotation_degrees;
          flag_is_wheel_animation_running = false;
        } else {
          variable_current_rotation_degrees +=
            variable_angular_gap * constant_clover_rotation_easing_factor;
        }
        function_render_wheel_transform();
        if (flag_is_wheel_animation_running) {
          window.requestAnimationFrame(function_ease_wheel_rotation_step);
        }
      };

      var function_on_scroll_update_wheel = function () {
        function_read_target_rotation_from_scroll();
        if (!flag_is_wheel_animation_running) {
          flag_is_wheel_animation_running = true;
          window.requestAnimationFrame(function_ease_wheel_rotation_step);
        }
      };

      function_read_target_rotation_from_scroll();
      variable_current_rotation_degrees = variable_target_rotation_degrees;
      function_render_wheel_transform();
      window.addEventListener('scroll', function_on_scroll_update_wheel, { passive: true });
      window.addEventListener('resize', function_on_scroll_update_wheel);
    }

    function_bind_real_world_palette_and_link_transition() {
      if (!this.variable_pinned_mark_container || !this.variable_first_real_world_section) {
        return;
      }
      var variable_leaf_anchor_elements = Array.prototype.slice.call(
        this.variable_pinned_mark_container.querySelectorAll('a.clover__leaf-hit[href]')
      );
      variable_leaf_anchor_elements.forEach(function (variable_leaf_anchor) {
        variable_leaf_anchor.setAttribute('data-stage-href', variable_leaf_anchor.getAttribute('href'));
        variable_leaf_anchor.setAttribute(
          'data-stage-label',
          variable_leaf_anchor.getAttribute('aria-label') || ''
        );
      });

      var function_repoint_leaf_anchor_targets = function () {
        var flag_is_real_world_active = document.documentElement.classList.contains('is-real-world');
        variable_leaf_anchor_elements.forEach(function (variable_leaf_anchor) {
          var variable_original_stage_href =
            variable_leaf_anchor.getAttribute('data-stage-href') || '';
          var variable_stage_slug = variable_original_stage_href.replace('#stage-', '');
          var variable_real_world_section = flag_is_real_world_active
            ? document.getElementById('real-world-' + variable_stage_slug)
            : null;
          if (variable_real_world_section) {
            var variable_section_heading = variable_real_world_section.querySelector('h2');
            variable_leaf_anchor.setAttribute('href', '#' + variable_real_world_section.id);
            if (variable_section_heading) {
              variable_leaf_anchor.setAttribute('aria-label', variable_section_heading.textContent);
            }
          } else {
            variable_leaf_anchor.setAttribute('href', variable_original_stage_href);
            variable_leaf_anchor.setAttribute(
              'aria-label',
              variable_leaf_anchor.getAttribute('data-stage-label')
            );
          }
        });
      };

      var variable_mark_svg_element = this.variable_pinned_mark_container.querySelector('svg');
      var variable_healthy_mark_aria_label = variable_mark_svg_element
        ? variable_mark_svg_element.getAttribute('aria-label') || ''
        : '';
      var variable_first_real_world_element = this.variable_first_real_world_section;
      var flag_was_in_real_world_half = null;

      var function_apply_real_world_state = function (flag_is_now_in_real_world) {
        if (flag_is_now_in_real_world === flag_was_in_real_world_half) {
          return;
        }
        flag_was_in_real_world_half = flag_is_now_in_real_world;
        document.documentElement.classList.toggle('is-real-world', flag_is_now_in_real_world);
        if (variable_mark_svg_element) {
          variable_mark_svg_element.setAttribute(
            'aria-label',
            flag_is_now_in_real_world
              ? constant_real_world_muted_clover_aria_label
              : variable_healthy_mark_aria_label
          );
        }
        function_repoint_leaf_anchor_targets();
      };

      var function_synchronize_real_world_boundary = function () {
        var flag_has_crossed_midpoint =
          variable_first_real_world_element.getBoundingClientRect().top <= window.innerHeight / 2;
        function_apply_real_world_state(flag_has_crossed_midpoint);
      };

      function_synchronize_real_world_boundary();
      window.addEventListener('scroll', function_synchronize_real_world_boundary, { passive: true });
      window.addEventListener('resize', function_synchronize_real_world_boundary);
    }

    function_bind_closing_section_visibility() {
      if (!this.variable_closing_answer_section) {
        return;
      }
      var variable_closing_element = this.variable_closing_answer_section;
      var function_synchronize_closing_state = function () {
        var flag_is_closing_section_visible =
          variable_closing_element.getBoundingClientRect().top <=
          window.innerHeight * constant_closing_section_viewport_ratio;
        document.documentElement.classList.toggle('is-closing', flag_is_closing_section_visible);
      };

      function_synchronize_closing_state();
      window.addEventListener('scroll', function_synchronize_closing_state, { passive: true });
      window.addEventListener('resize', function_synchronize_closing_state);
    }
  }

  // ==========================================================================
  // 3. `class_site_navigation_drawer_controller` — Responsive Header Menu
  // ==========================================================================

  class class_site_navigation_drawer_controller {
    constructor() {
      this.variable_toggle_button_element = document.querySelector('.nav-toggle');
      this.variable_navigation_element = document.getElementById('site-nav');
    }

    function_initialize_navigation_drawer() {
      if (!this.variable_toggle_button_element || !this.variable_navigation_element) {
        return;
      }
      var variable_toggle_button = this.variable_toggle_button_element;
      var variable_navigation_bar = this.variable_navigation_element;

      var function_set_navigation_open_state = function (flag_should_open_menu) {
        variable_toggle_button.setAttribute('aria-expanded', flag_should_open_menu ? 'true' : 'false');
        variable_navigation_bar.classList.toggle('is-open', flag_should_open_menu);
      };

      variable_toggle_button.addEventListener('click', function () {
        var flag_is_currently_expanded =
          variable_toggle_button.getAttribute('aria-expanded') === 'true';
        function_set_navigation_open_state(!flag_is_currently_expanded);
      });

      variable_navigation_bar.addEventListener('click', function (variable_click_event) {
        if (variable_click_event.target.closest('a')) {
          function_set_navigation_open_state(false);
        }
      });

      document.addEventListener('keydown', function (variable_key_event) {
        var flag_is_escape_while_open =
          variable_key_event.key === 'Escape' &&
          variable_toggle_button.getAttribute('aria-expanded') === 'true';
        if (flag_is_escape_while_open) {
          function_set_navigation_open_state(false);
          variable_toggle_button.focus();
        }
      });

      window.addEventListener('resize', function () {
        if (window.innerWidth > constant_desktop_navigation_breakpoint_pixels) {
          function_set_navigation_open_state(false);
        }
      });
    }
  }

  // ==========================================================================
  // 4. `class_accessible_tab_and_glossary_controller` — Tabs (`start/`) & Glossary Filter (`glossary/`)
  // ==========================================================================

  class class_accessible_tab_and_glossary_controller {
    constructor() {
      this.variable_tab_root_elements = Array.prototype.slice.call(
        document.querySelectorAll('[data-tabs]')
      );
      this.variable_glossary_filter_input = document.getElementById('term-filter');
      this.variable_glossary_row_elements = Array.prototype.slice.call(
        document.querySelectorAll('.terms > div')
      );
      this.variable_glossary_count_status = document.getElementById('term-count');
    }

    function_initialize_tabs_and_glossary() {
      var variable_self_controller = this;
      this.variable_tab_root_elements.forEach(function (variable_tab_root) {
        variable_self_controller.function_bind_accessible_tab_group(variable_tab_root);
      });
      this.function_bind_glossary_live_filter();
      this.function_annotate_responsive_data_table_labels();
    }

    function_annotate_responsive_data_table_labels() {
      var variable_data_tables = Array.prototype.slice.call(
        document.querySelectorAll('table.data')
      );
      variable_data_tables.forEach(function (variable_table) {
        var variable_header_cells = Array.prototype.slice.call(
          variable_table.querySelectorAll('thead th')
        );
        if (!variable_header_cells.length) {
          return;
        }
        var variable_header_labels = variable_header_cells.map(function (variable_th) {
          var variable_clone = variable_th.cloneNode(true);
          Array.prototype.slice
            .call(variable_clone.querySelectorAll('br'))
            .forEach(function (variable_br) {
              variable_br.parentNode.replaceChild(document.createTextNode(' '), variable_br);
            });
          return (variable_clone.textContent || '').replace(/\s+/g, ' ').trim();
        });
        var variable_body_rows = Array.prototype.slice.call(
          variable_table.querySelectorAll('tbody tr')
        );
        variable_body_rows.forEach(function (variable_row) {
          var variable_cells = Array.prototype.slice.call(variable_row.children);
          variable_cells.forEach(function (variable_cell, variable_col_index) {
            if (
              variable_col_index > 0 &&
              variable_header_labels[variable_col_index] &&
              !variable_cell.hasAttribute('data-label')
            ) {
              variable_cell.setAttribute('data-label', variable_header_labels[variable_col_index]);
            }
          });
        });
      });

      function function_sync_stacked_tables() {
        variable_data_tables.forEach(function (variable_table) {
          var variable_wrap = variable_table.closest('.table-wrap');
          if (!variable_wrap) {
            return;
          }
          if (window.innerWidth <= 960) {
            variable_wrap.classList.add('is-stacked');
            return;
          }
          variable_wrap.classList.remove('is-stacked');
          if (variable_wrap.scrollWidth > variable_wrap.clientWidth + 4) {
            variable_wrap.classList.add('is-stacked');
          }
        });
      }

      function_sync_stacked_tables();
      window.addEventListener('resize', function_sync_stacked_tables);
    }

    function_select_active_tab_panel(
      variable_tab_buttons,
      variable_tab_panels,
      variable_selected_index,
      flag_should_focus_tab
    ) {
      variable_tab_buttons.forEach(function (variable_tab_button, variable_current_index) {
        var flag_is_selected_tab = variable_current_index === variable_selected_index;
        variable_tab_button.setAttribute('aria-selected', flag_is_selected_tab ? 'true' : 'false');
        variable_tab_button.setAttribute('tabindex', flag_is_selected_tab ? '0' : '-1');
        variable_tab_panels[variable_current_index].classList.toggle('is-active', flag_is_selected_tab);
        variable_tab_panels[variable_current_index].hidden = !flag_is_selected_tab;
      });
      if (flag_should_focus_tab) {
        variable_tab_buttons[variable_selected_index].focus();
      }
    }

    function_bind_accessible_tab_group(variable_tab_root_element) {
      var variable_tablist_element = variable_tab_root_element.querySelector('[role="tablist"]');
      var variable_tab_buttons = Array.prototype.slice.call(
        variable_tab_root_element.querySelectorAll('[role="tab"]')
      );
      var variable_tab_panels = variable_tab_buttons.map(function (variable_tab_button) {
        return document.getElementById(variable_tab_button.getAttribute('aria-controls'));
      });
      if (
        !variable_tablist_element ||
        !variable_tab_buttons.length ||
        variable_tab_panels.indexOf(null) !== -1
      ) {
        return;
      }

      var variable_self_controller = this;
      var function_activate_tab_at_index = function (variable_target_index, flag_should_focus_tab) {
        variable_self_controller.function_select_active_tab_panel(
          variable_tab_buttons,
          variable_tab_panels,
          variable_target_index,
          flag_should_focus_tab
        );
      };

      variable_tab_buttons.forEach(function (variable_tab_button, variable_button_index) {
        variable_tab_button.addEventListener('click', function () {
          function_activate_tab_at_index(variable_button_index, false);
        });
        variable_tab_button.addEventListener('keydown', function (variable_key_event) {
          var flag_is_vertical_tablist =
            variable_tablist_element.getAttribute('aria-orientation') === 'vertical';
          var variable_direction_step = {
            ArrowRight: 1,
            ArrowDown: 1,
            ArrowLeft: -1,
            ArrowUp: -1
          }[variable_key_event.key];
          var flag_matches_orientation_arrow =
            variable_direction_step &&
            ((flag_is_vertical_tablist &&
              (variable_key_event.key === 'ArrowUp' || variable_key_event.key === 'ArrowDown')) ||
              (!flag_is_vertical_tablist &&
                (variable_key_event.key === 'ArrowLeft' || variable_key_event.key === 'ArrowRight')));

          if (flag_matches_orientation_arrow) {
            variable_key_event.preventDefault();
            var variable_wrapped_index =
              (variable_button_index + variable_direction_step + variable_tab_buttons.length) %
              variable_tab_buttons.length;
            function_activate_tab_at_index(variable_wrapped_index, true);
          } else if (variable_key_event.key === 'Home') {
            variable_key_event.preventDefault();
            function_activate_tab_at_index(0, true);
          } else if (variable_key_event.key === 'End') {
            variable_key_event.preventDefault();
            function_activate_tab_at_index(variable_tab_buttons.length - 1, true);
          }
        });
      });

      var variable_initial_tab_index = variable_tab_buttons.findIndex(function (variable_tab_button) {
        return variable_tab_button.getAttribute('aria-selected') === 'true';
      });
      function_activate_tab_at_index(
        variable_initial_tab_index === -1 ? 0 : variable_initial_tab_index,
        false
      );

      var function_activate_tab_matching_url_hash = function () {
        var variable_hash_target_id = window.location.hash.slice(1);
        var variable_matching_tab_index = variable_tab_buttons.findIndex(function (
          variable_tab_button
        ) {
          return variable_tab_button.id === variable_hash_target_id;
        });
        if (variable_matching_tab_index !== -1) {
          function_activate_tab_at_index(variable_matching_tab_index, false);
        }
      };
      function_activate_tab_matching_url_hash();
      window.addEventListener('hashchange', function_activate_tab_matching_url_hash);
    }

    function_bind_glossary_live_filter() {
      if (!this.variable_glossary_filter_input) {
        return;
      }
      var variable_filter_input = this.variable_glossary_filter_input;
      var variable_term_rows = this.variable_glossary_row_elements;
      var variable_count_status_element = this.variable_glossary_count_status;

      variable_filter_input.addEventListener('input', function () {
        var variable_normalized_query = variable_filter_input.value.trim().toLowerCase();
        var variable_visible_term_count = 0;
        variable_term_rows.forEach(function (variable_term_row) {
          var flag_row_matches_query =
            !variable_normalized_query ||
            variable_term_row.textContent.toLowerCase().indexOf(variable_normalized_query) !== -1;
          variable_term_row.hidden = !flag_row_matches_query;
          if (flag_row_matches_query) {
            variable_visible_term_count++;
          }
        });
        if (variable_count_status_element) {
          variable_count_status_element.textContent = variable_normalized_query
            ? variable_visible_term_count +
              (variable_visible_term_count === 1 ? ' term matches' : ' terms match') +
              ' "' +
              variable_filter_input.value.trim() +
              '"'
            : '';
        }
      });
    }
  }

  // ==========================================================================
  // BOOTSTRAP — Instantiate and wire the 4 Website Domain Controllers
  // ==========================================================================

  var variable_viewport_scroll_navigator = new class_viewport_scroll_navigator();
  variable_viewport_scroll_navigator.function_bind_document_anchor_click_delegation();

  var variable_five_stage_clover_mark_controller = new class_five_stage_clover_mark_controller(
    variable_viewport_scroll_navigator.flag_prefers_reduced_motion
  );
  variable_five_stage_clover_mark_controller.function_initialize_stage_highlighting_and_wheel_rotation();

  var variable_site_navigation_drawer_controller = new class_site_navigation_drawer_controller();
  variable_site_navigation_drawer_controller.function_initialize_navigation_drawer();

  var variable_accessible_tab_and_glossary_controller =
    new class_accessible_tab_and_glossary_controller();
  variable_accessible_tab_and_glossary_controller.function_initialize_tabs_and_glossary();
})();
