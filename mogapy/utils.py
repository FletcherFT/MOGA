import csv
import itertools
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter

# TODO For some reason, ffmpeg_path can't be found by shutil.which("ffmpeg")
plt.rcParams[
    'animation.ffmpeg_path'] = 'C:\\Users\\fletho\\ffmpeg-4.2.2-win64-static\\ffmpeg-4.2.2-win64-static\\bin\\ffmpeg'


class ResultsManager(FFMpegWriter):
    def __init__(self, **kwargs):
        self._F = None
        self._axes = None
        self._log = None
        self._plots = []
        self._i = 0
        self.flag = False
        if len(kwargs):
            self._fname = kwargs.pop("outfile")
            self.flag = True
            super().__init__(**kwargs)

    def update(self, A, names, **kwargs):
        _, numvars = A.shape
        self._i += 1
        names.reverse()
        if self._F is None:
            self._F, self._axes = plt.subplots(nrows=numvars, ncols=numvars, figsize=(8, 8))
            self._F.suptitle("Generation {:15d}".format(self._i))
            self._F.subplots_adjust(hspace=0.05, wspace=0.05)
            if numvars > 1:
                for ax in self._axes.flat:
                    # Hide all ticks and labels
                    ax.xaxis.set_visible(False)
                    ax.yaxis.set_visible(False)
                    # Set up ticks only on one side for the "edge" subplots...
                    if ax.is_first_col():
                        ax.yaxis.set_ticks_position('left')
                    if ax.is_last_col():
                        ax.yaxis.set_ticks_position('right')
                    if ax.is_first_row():
                        ax.xaxis.set_ticks_position('top')
                    if ax.is_last_row():
                        ax.xaxis.set_ticks_position('bottom')
                # Label the diagonal subplots...
                for i, label in enumerate(names):
                    self._axes[i, i].annotate(label, (0.5, 0.5), xycoords='axes fraction',
                                              ha='center', va='center')
                # Turn on the proper x or y axes ticks.
                for i, j in zip(range(numvars), itertools.cycle((-1, 0))):
                    self._axes[j, i].xaxis.set_visible(True)
                    self._axes[i, j].yaxis.set_visible(True)
                # Plot the data.
                for i, j in zip(*np.triu_indices_from(self._axes, k=1)):
                    for x, y in [(i, j), (j, i)]:
                        p = self._axes[x, y].plot(A[:, x], A[:, y], **kwargs)
                        self._plots += p
                if self.flag:
                    self.setup(self._F, outfile=self._fname)
            else:
                self._plots = self._axes.plot(A, A, **kwargs)
        else:
            c = 0
            if numvars > 1:
                for i, j in zip(*np.triu_indices_from(self._axes, k=1)):
                    for x, y in [(i, j), (j, i)]:
                        self._plots[c].set_xdata(A[:, x])
                        self._plots[c].set_ydata(A[:, y])
                        self._plots[c].axes.set_xlim(xmin=A[:, x].min(), xmax=A[:, x].max())
                        self._plots[c].axes.set_ylim(ymin=A[:, y].min(), ymax=A[:, y].max())
                        c += 1
            else:
                self._plots[0].set_xdata(A)
                self._plots[0].set_ydata(A)
                self._plots[0].axes.set_xlim(xmin=A.min(), xmax=A.max())
                self._plots[0].axes.set_ylim(ymin=A.min(), ymax=A.max())
            self._F.suptitle("Generation {:15d}".format(self._i))
        self._F.canvas.draw()
        self._F.canvas.flush_events()
        if self.flag:
            self.grab_frame()

    def log(self, A, names):
        # TODO Find something worthwhile to log.
        raise NotImplementedError("Still Working On This.")
        if self._log is None:
            logdir = Path("./results").resolve()
            logdir.mkdir(parents=True, exist_ok=True)
            numruns = len(list(logdir.glob("run*")))
            fname = logdir.joinpath("run_{:04d}".format(numruns + 1))
            fname.mkdir(parents=True, exist_ok=True)
            fname.joinpath("log.csv")
            self._log = str(fname)
            with open(self._log, "a") as f:
                writer = csv.writer(f, delimeter=',')
                writer.writerow(names)
                writer.writerow(A)
            return
        with open(self._log, "a") as f:
            writer = csv.writer(f, delimeter=',')
            writer.writerow(A)


class ResultPlotter(FFMpegWriter):
    def __init__(self, n, cost, **kwargs):
        self._F = None
        self._ax = None
        self._p = []
        self._i = 0
        self.flag = False
        self._n = n
        if len(kwargs):
            self._fname = kwargs.pop("outfile")
            self.flag = True
            super().__init__(**kwargs)
        self._F, self._ax = plt.subplots()
        self._ax.quiver(cost[:, :, 0], cost[:, :, 1], cost[:, :, 2], cost[:, :, 3])
        self._L = cost.shape[0]
        for _ in range(n):
            self._p += self._ax.plot(0,0)
        self._ax.set_xlabel("Horizontal Position")
        self._ax.set_ylabel("Altitude from Sea Bottom")
        if self.flag:
            self.setup(self._F, outfile=self._fname)

    def update(self, solutions):
        self._i += 1
        if self._F is None:
            self._F, self._ax = plt.subplots()
            self._ax.quiver(cost[:, :, 0], cost[:, :, 1], cost[:, :, 2], cost[:, :, 3])
            for solution in solutions:
                self._p += plt.plot(solution[:, 0], solution[:, 1])
            if self.flag:
                self.setup(self._F, outfile=self._fname)
            self._ax.set_xlabel("Horizontal Position")
            self._ax.set_ylabel("Altitude from Sea Bottom")
        else:
            for i, solution in enumerate(solutions):
                self._p[i].set_xdata(solution[:, 0])
                self._p[i].set_ydata(solution[:, 1])
        self._F.suptitle("Generation {:15d}".format(self._i))
        self._F.canvas.draw()
        self._F.canvas.flush_events()
        if self.flag:
            self.grab_frame()
