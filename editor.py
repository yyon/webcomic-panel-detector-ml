import gi
import random
import cairo
import os
import json

from gi.repository import GLib

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, Gdk, GdkPixbuf, Gio

COLORS = [ \
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (0, 1, 1),
    (1, 0, 1),
]

BOX_WIDTH = 30
PADDING = 2
COLUMNS = 5

class ImageViewer(Gtk.Application):
    def __init__(self):
        super().__init__(application_id="org.example.ImageViewer")
        self.connect("activate", self.on_activate)

        # Hard-coded folder path
        self.image_folder = "/home/yyon/Pictures/long_panels/unsolved"
        self.image_files = []
        self.current_image_index = -1
        
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0

        self.regions = []
        self.current_region_start = None
        self.current_region_temp = None

        self.last_drag_x = 0
        self.last_drag_y = 0

        self.last_pointer_position = None

    def on_activate(self, app):
        self.window = Gtk.ApplicationWindow(application=app)
        self.window.set_title("Image Viewer")
        self.window.set_default_size(800, 600)
        self.window.maximize()

        # Create main box
        self.main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.window.set_child(self.main_box)

        # Create toolbar
        self.toolbar = Gtk.Box(spacing=5, margin_top=5, margin_bottom=5, margin_start=5, margin_end=5)
        self.main_box.append(self.toolbar)

        # Add navigation buttons
        self.prev_button = Gtk.Button(label="← Previous")
        self.prev_button.connect("clicked", self.on_prev_clicked)
        self.toolbar.append(self.prev_button)

        self.next_button = Gtk.Button(label="Next →")
        self.next_button.connect("clicked", self.on_next_clicked)
        self.toolbar.append(self.next_button)

        # Add separator
        self.toolbar.append(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL))

        # Add status label
        self.status_label = Gtk.Label()
        self.toolbar.append(self.status_label)

        # Add spacer
        spacer = Gtk.Box()
        spacer.set_hexpand(True)
        self.toolbar.append(spacer)

        # clear
        self.next_button = Gtk.Button(label="Set Single Regions")
        self.next_button.connect("clicked", self.on_single_region_clicked)
        self.toolbar.append(self.next_button)

        # clear
        self.next_button = Gtk.Button(label="Clear Regions")
        self.next_button.connect("clicked", self.on_clear_regions_clicked)
        self.toolbar.append(self.next_button)

        # Create scrolled window for image
        self.scrolled = Gtk.ScrolledWindow()
        self.scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.NEVER)
        self.scrolled.set_vexpand(True)
        self.main_box.append(self.scrolled)

        # Create drawing area
        self.drawing_area = Gtk.DrawingArea()
        self.drawing_area.set_draw_func(self.on_draw)
        self.scrolled.set_child(self.drawing_area)

        # Load image files from folder
        self.load_image_files()
        
        # Load first image if available
        if self.image_files:
            GLib.timeout_add(50, self.load_first_image)

        self.window.present()

        # Add gestures and controllers

        zoom_gesture = Gtk.GestureZoom.new()
        zoom_gesture.connect("scale-changed", self.on_zoom)
        self.drawing_area.add_controller(zoom_gesture)

        drag_gesture = Gtk.GestureDrag.new()
        drag_gesture.set_button(3)
        drag_gesture.connect("drag-begin", self.on_drag_begin)
        drag_gesture.connect("drag-update", self.on_drag_update)
        self.drawing_area.add_controller(drag_gesture)

        click_gesture = Gtk.GestureClick.new()
        click_gesture.set_button(1)
        click_gesture.connect("pressed", self.on_click_start)
        click_gesture.connect("released", self.on_click_end)
        self.drawing_area.add_controller(click_gesture)

        motion_controller = Gtk.EventControllerMotion.new()
        motion_controller.connect("motion", self.on_mouse_motion)
        self.drawing_area.add_controller(motion_controller)

        scroll_controller = Gtk.EventControllerScroll.new(Gtk.EventControllerScrollFlags.VERTICAL)
        scroll_controller.connect("scroll", self.on_scroll)
        self.drawing_area.add_controller(scroll_controller)

    def load_image_files(self):
        try:
            self.image_files = [
                os.path.join(self.image_folder, f)
                for f in os.listdir(self.image_folder)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))
            ]
            self.image_files.sort()
        except Exception as e:
            print(f"Error loading image files: {e}")
            self.image_files = []

    def load_first_image(self):
        self.load_image_index(0)
        return False

    def load_image_index(self, index):
        if not self.image_files or index < 0 or index >= len(self.image_files):
            print("error")
            return False
            
        self.current_image_index = index
        image_file = self.image_files[index]
        self.load_image(image_file)
        self.update_status()
        
        # Update buttons state
        self.prev_button.set_sensitive(self.current_image_index > 0)
        self.next_button.set_sensitive(self.current_image_index < len(self.image_files) - 1)
        
        # Reset view and regions
        self.regions = []
        self.current_region_start = None
        self.current_region_temp = None
        self.init_scale()

        # load regions from json
        json_file = image_file.rsplit(".", 1)[0] + ".json"
        if os.path.exists(json_file):
            with open(json_file, "r") as f:
                json_text = f.read()
                loaded_regions = json.loads(json_text)
                self.regions = [{'y1': region[0], 'y2': region[0] + region[1]} for region in loaded_regions]

        return True

    def update_status(self):
        if self.image_files and 0 <= self.current_image_index < len(self.image_files):
            self.status_label.set_text(
                f"Image {self.current_image_index + 1} of {len(self.image_files)}: "
                f"{os.path.basename(self.image_files[self.current_image_index])}"
            )
        else:
            self.status_label.set_text("No images available")

    def on_prev_clicked(self, button):
        if self.current_image_index > 0:
            self.load_image_index(self.current_image_index - 1)

    def on_next_clicked(self, button):
        if self.current_image_index < len(self.image_files) - 1:
            self.load_image_index(self.current_image_index + 1)

    def on_single_region_clicked(self, button):
        img_height = self.pixbuf.get_height()
        self.regions = [{'y1': 0, 'y2': img_height}]
        self.current_region_start = None
        self.current_region_temp = None
        self.save_regions()
        self.drawing_area.queue_draw()

    def on_clear_regions_clicked(self, button):
        self.regions = []
        self.current_region_start = None
        self.current_region_temp = None
        self.drawing_area.queue_draw()

    def init_scale(self):
        if not hasattr(self, "pixbuf"):
            return False
            
        alloc = self.drawing_area.get_allocation()
        win_width = alloc.width
        win_height = alloc.height
        img_width = self.pixbuf.get_width()
        img_height = self.pixbuf.get_height()
        self.scale = min(win_width / img_width, win_height / img_height) * 0.95
        self.offset_x = (win_width - (img_width * self.scale)) / 2
        self.offset_y = (win_height - (img_height * self.scale)) / 2

        self.drawing_area.queue_draw()
        return False

    def load_image(self, path):
        self.pixbuf = GdkPixbuf.Pixbuf.new_from_file(path)

    def on_draw(self, area, ctx, width, height):
        if not hasattr(self, "pixbuf"):
            return

        ctx.save()
        ctx.translate(self.offset_x, self.offset_y)
        ctx.scale(self.scale, self.scale)

        Gdk.cairo_set_source_pixbuf(ctx, self.pixbuf, 0, 0)
        ctx.paint()

        image_width = self.pixbuf.get_width()
        image_height = self.pixbuf.get_height()

        all_regions = self.regions[:]
        temp_region = None
        if self.current_region_temp:
            y1, y2 = self.current_region_temp
            temp_region = {'y1': y1, 'y2': y2}
            all_regions.append(temp_region)
            all_regions.sort(key=lambda x : x['y1'])

        for i, region in enumerate(all_regions):
            ctx.set_source_rgba(0,0,0,1)
            ctx.set_line_width(4.0 / self.scale)
            ctx.move_to(0, region['y1'])
            ctx.line_to(image_width, region['y1'])
            ctx.stroke()
            ctx.move_to(0, region['y2'])
            ctx.line_to(image_width, region['y2'])
            ctx.stroke()
            ctx.set_source_rgba(*COLORS[i % len(COLORS)], 0.5 if region==temp_region else 1)
            ctx.set_line_width(2.0 / self.scale)
            ctx.move_to(0, region['y1'])
            ctx.line_to(image_width, region['y1'])
            ctx.stroke()
            ctx.move_to(0, region['y2'])
            ctx.line_to(image_width, region['y2'])
            ctx.stroke()

        ctx.restore()

        for i, region in enumerate(all_regions):
            col = i % COLUMNS
            x = image_width * self.scale + self.offset_x + ((col + 1) * (BOX_WIDTH + PADDING))
            y1 = region['y1'] * self.scale + self.offset_y
            y2 = region['y2'] * self.scale + self.offset_y
            h = abs(y2 - y1)

            ctx.set_source_rgba(*COLORS[i % len(COLORS)], 0.5 if region==temp_region else 1)
            ctx.rectangle(x, min(y1, y2), BOX_WIDTH, h)
            ctx.fill()

    def on_zoom(self, gesture, scale):
        center_x = self.drawing_area.get_allocated_width() / 2
        center_y =  self.drawing_area.get_allocated_height() / 2

        image_x = (center_x - self.offset_x) / self.scale
        image_y = (center_y - self.offset_y) / self.scale

        self.scale *= scale

        self.offset_x = center_x - image_x * self.scale
        self.offset_y = center_y - image_y * self.scale

        self.drawing_area.queue_draw()

    def on_scroll(self, controller, dx, dy):
        if dy != 0:
            zoom_factor = 1.1 if dy < 0 else 0.9

            center_x = self.drawing_area.get_allocated_width() / 2
            center_y = self.drawing_area.get_allocated_height() / 2

            if self.last_pointer_position != None:
                center_x, center_y = self.last_pointer_position

            image_x = (center_x - self.offset_x) / self.scale
            image_y = (center_y - self.offset_y) / self.scale

            self.scale *= zoom_factor

            self.offset_x = center_x - image_x * self.scale
            self.offset_y = center_y - image_y * self.scale

            self.drawing_area.queue_draw()
        return True

    def on_drag_begin(self, gesture, start_x, start_y):
        self.last_drag_x = 0
        self.last_drag_y = 0

    def on_drag_update(self, gesture, offset_x, offset_y):
        dx = offset_x - self.last_drag_x
        dy = offset_y - self.last_drag_y
        self.last_drag_x = offset_x
        self.last_drag_y = offset_y
        self.offset_x += dx
        self.offset_y += dy
        self.drawing_area.queue_draw()

    def on_click_start(self, gesture, n_press, x, y):
        mouse_y = (y - self.offset_y) / self.scale

        if self.current_region_start != None and self.current_region_temp != None:
            self.finish_region(mouse_y)
            return
        
        img_height = self.pixbuf.get_height()
        mouse_y = min(max(mouse_y, 0), img_height)
        self.current_region_start = mouse_y
        image_width = self.pixbuf.get_width()
        for i, region in enumerate(self.regions):
            grabbed_top = abs(mouse_y-region["y1"]) < 10
            grabbed_bottom = abs(mouse_y-region["y2"]) < 10
            if grabbed_top or grabbed_bottom:
                col = i % COLUMNS
                box_x = image_width * self.scale + self.offset_x + ((col + 1) * (BOX_WIDTH + PADDING))
                if x >= box_x and x < box_x + BOX_WIDTH:
                    self.regions.remove(region)
                    if grabbed_top:
                        self.current_region_start = region["y2"]
                    else:
                        self.current_region_start = region["y1"]
                    break
        self.current_region_temp = (self.current_region_start, mouse_y)
        self.drawing_area.queue_draw()

    def on_mouse_motion(self, controller, x, y):
        self.last_pointer_position = [x,y]
        if self.current_region_start is not None:
            y_current = (y - self.offset_y) / self.scale
            img_height = self.pixbuf.get_height()
            y_current = min(max(y_current, 0), img_height)
            self.current_region_temp = (self.current_region_start, y_current)
            self.drawing_area.queue_draw()

    def on_click_end(self, gesture, n_press, x, y):
        if self.current_region_start != None:
            mouse_y = (y - self.offset_y) / self.scale
            img_height = self.pixbuf.get_height()
            y1 = min(max(min(self.current_region_start, mouse_y), 0), img_height)
            y2 = min(max(max(self.current_region_start, mouse_y), 0), img_height)
            if abs(y1-y2) > 5:
                self.finish_region(mouse_y)

    def finish_region(self, y_end):
        img_height = self.pixbuf.get_height()
        y1 = min(max(min(self.current_region_start, y_end), 0), img_height)
        y2 = min(max(max(self.current_region_start, y_end), 0), img_height)

        if abs(y1-y2) < 5:
            y1 = 0
            for region in self.regions:
                if region['y2'] < y2:
                    y1 = region['y2']

        self.regions.append({'y1': y1, 'y2': y2})
        self.regions.sort(key=lambda x : x['y1'])
        self.save_regions()
        self.current_region_start = None
        self.current_region_temp = None
        self.drawing_area.queue_draw()
    
    def save_regions(self):
        regions_arr = [[region['y1'], region['y2'] - region['y1']] for region in self.regions]

        image_file = self.image_files[self.current_image_index]
        json_file = image_file.rsplit(".", 1)[0] + ".json"
        with open(json_file, "w") as f:
            json_text = f.write(json.dumps(regions_arr))


app = ImageViewer()
app.run()