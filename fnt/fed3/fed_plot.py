"""FED3 plot rendering and management.

Separates matplotlib rendering logic from the main widget.
"""

from datetime import datetime, timedelta
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

class FedPlotManager:
    def __init__(self, canvas, ax, plot_placeholder):
        self.canvas = canvas
        self.ax = ax
        self.plot_placeholder = plot_placeholder

    def update_plot(self, target_dev, all_devices):
        self.ax.clear()
        self.ax.set_title("Pellets retrieved", color='white')
        self.ax.set_xlabel("Time", color='white')
        self.ax.set_ylabel("Cumulative Pellets", color='white')
        self.ax.tick_params(colors='white')
        for spine in self.ax.spines.values():
            spine.set_edgecolor('#444444')
        
        plotted_something = False
        min_time = None
        max_time = None
        
        devices_to_plot = [target_dev] if target_dev else all_devices
        
        for dev in devices_to_plot:
            events = dev.get('events', [])
            dev_name = dev['name_edit'].text().strip() or dev['box'].title()
            
            if events:
                times = mdates.date2num(events)
                counts = list(range(1, len(events) + 1))
                
                if min_time is None or events[0] < min_time:
                    min_time = events[0]
                if max_time is None or events[-1] > max_time:
                    max_time = events[-1]
                
                # Use standard .plot() instead of deprecated/removed .plot_date()
                self.ax.plot(times, counts, '-', label=dev_name, linewidth=2, marker='o', markersize=6, drawstyle='steps-post')
                plotted_something = True
            elif dev.get('is_tracking') and dev.get('tracking_start_time'):
                start_t = dev['tracking_start_time']
                end_t = max(datetime.now(), start_t + timedelta(minutes=5))
                times = mdates.date2num([start_t, end_t])
                counts = [0, 0]
                
                if min_time is None or start_t < min_time:
                    min_time = start_t
                if max_time is None or end_t > max_time:
                    max_time = end_t
                
                # Use standard .plot() instead of deprecated/removed .plot_date()
                self.ax.plot(times, counts, '-', label=dev_name, linewidth=2, marker='o', markersize=6, drawstyle='steps-post')
                plotted_something = True
        
        # Toggle between placeholder and canvas
        self.plot_placeholder.setVisible(not plotted_something)
        self.canvas.setVisible(plotted_something)
                
        if plotted_something:
            self.ax.legend(facecolor='#2b2b2b', edgecolor='#444444', labelcolor='white')
            
            # Shade dark cycle (19:00 to 07:00)
            if min_time and max_time:
                current_shade_start = min_time.replace(hour=19, minute=0, second=0, microsecond=0)
                if min_time.hour < 19:
                    current_shade_start -= timedelta(days=1)
                
                while current_shade_start < max_time:
                    shade_end = current_shade_start + timedelta(hours=12)
                    self.ax.axvspan(mdates.date2num(current_shade_start), 
                                    mdates.date2num(shade_end), 
                                    color='gray', alpha=0.3, zorder=0)
                    current_shade_start += timedelta(days=1)
            
            # Auto-scale x-axis so ticks adapt as the time range grows
            locator = mdates.AutoDateLocator()
            self.ax.xaxis.set_major_locator(locator)
            self.ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
            self.ax.autoscale_view()
            
            # Overwrite autoscaled limits with safer bounds to prevent zero-range formatting errors
            if min_time and max_time:
                if (max_time - min_time).total_seconds() < 300:
                    mid = min_time + (max_time - min_time) / 2
                    min_t_plot = mid - timedelta(minutes=2.5)
                    max_t_plot = mid + timedelta(minutes=2.5)
                else:
                    min_t_plot = min_time
                    max_t_plot = max_time
                self.ax.set_xlim(mdates.date2num(min_t_plot), mdates.date2num(max_t_plot))
            
            # Configure integer ticks on y-axis
            self.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            
            max_y = 0
            for dev in devices_to_plot:
                events = dev.get('events', [])
                if events:
                    max_y = max(max_y, len(events))
            if max_y == 0:
                self.ax.set_ylim(0, 10)
            else:
                self.ax.set_ylim(bottom=0)
                
            self.canvas.figure.autofmt_xdate()
            self.canvas.figure.tight_layout(pad=1.5)
            self.canvas.draw()
