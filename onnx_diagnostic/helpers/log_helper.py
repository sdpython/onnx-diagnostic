import datetime
import glob
import os
import re
import zipfile
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas
from pandas.api.types import is_numeric_dtype
from .helper import string_sig


def enumerate_csv_files(
    data: Union[
        pandas.DataFrame, List[Union[str, Tuple[str, str]]], str, Tuple[str, str, str, str]
    ],
    verbose: int = 0,
) -> Iterator[Union[pandas.DataFrame, str, Tuple[str, str, str, str]]]:
    """
    Enumerates files considered for the aggregation.
    Only csv files are considered.
    If a zip file is given, the function digs into the zip files and
    loops over csv candidates.

    :param data: dataframe with the raw data or a file or list of files

    data can contains:
    * a dataframe
    * a string for a filename, zip or csv
    * a list of string
    * a tuple
    """
    if not isinstance(data, list):
        data = [data]
    for itn, filename in enumerate(data):
        if isinstance(filename, pandas.DataFrame):
            if verbose:
                print(f"[enumerate_csv_files] data[{itn}] is a dataframe")
            yield filename
            continue

        if isinstance(filename, tuple):
            # A file in a zipfile
            if verbose:
                print(f"[enumerate_csv_files] data[{itn}] is {filename!r}")
            yield filename
            continue

        if os.path.exists(filename):
            ext = os.path.splitext(filename)[-1]
            if ext == ".csv":
                # We check the first line is ok.
                if verbose:
                    print(f"[enumerate_csv_files] data[{itn}] is a csv file: {filename!r}]")
                with open(filename, "r", encoding="utf-8") as f:
                    line = f.readline()
                    if "~help" in line or (",CMD" not in line and ",DATE" not in line):
                        continue
                    dt = datetime.datetime.fromtimestamp(os.stat(filename).st_mtime)
                    du = dt.strftime("%Y-%m-%d %H:%M:%S")
                    yield (os.path.split(filename)[-1], du, filename, "")
                continue

            if ext == ".zip":
                if verbose:
                    print(f"[enumerate_csv_files] data[{itn}] is a zip file: {filename!r}]")
                zf = zipfile.ZipFile(filename, "r")
                for ii, info in enumerate(zf.infolist()):
                    name = info.filename
                    ext = os.path.splitext(name)[-1]
                    if ext != ".csv":
                        continue
                    if verbose:
                        print(
                            f"[enumerate_csv_files] data[{itn}][{ii}] is a csv file: {name!r}]"
                        )
                    with zf.open(name) as zzf:
                        line = zzf.readline()
                    yield (
                        os.path.split(name)[-1],
                        "%04d-%02d-%02d %02d:%02d:%02d" % info.date_time,
                        name,
                        filename,
                    )
                zf.close()
                continue

            raise AssertionError(f"Unexpected format {filename!r}, cannot read it.")

        # filename is a pattern.
        found = glob.glob(filename)
        if verbose and not found:
            print(f"[enumerate_csv_files] unable to find file in {filename!r}")
        for ii, f in enumerate(found):
            if verbose:
                print(f"[enumerate_csv_files] data[{itn}][{ii}] {f!r} from {filename!r}")
            yield from enumerate_csv_files(f, verbose=verbose)


def open_dataframe(
    data: Union[str, Tuple[str, str, str, str], pandas.DataFrame],
) -> pandas.DataFrame:
    """
    Opens a filename.

    :param data: a dataframe, a filename, a tuple indicating the file is coming
        from a zip file
    :return: a dataframe
    """
    if isinstance(data, pandas.DataFrame):
        return data
    if isinstance(data, str):
        df = pandas.read_csv(data)
        df["RAWFILENAME"] = data
        return df
    if isinstance(data, tuple):
        if not data[-1]:
            df = pandas.read_csv(data[2])
            df["RAWFILENAME"] = data[2]
            return df
        zf = zipfile.ZipFile(data[-1])
        with zf.open(data[2]) as f:
            df = pandas.read_csv(f)
            df["RAWFILENAME"] = f"{data[-1]}/{data[2]}"
        zf.close()
        return df

    raise ValueError(f"Unexpected value for data: {data!r}")


class CubeViewDef:
    """
    Defines how to compute a view.

    :param key_index: keys to put in the row index
    :param values: values to show
    :param ignore_unique: ignore keys with a unique value
    :param order: to reorder key in columns index
    :param key_agg: aggregate according to these columns before
        creating the view
    :param agg_args: see :meth:`pandas.core.groupby.DataFrameGroupBy.agg`
    :param agg_kwargs: see :meth:`pandas.core.groupby.DataFrameGroupBy.agg`
    """

    def __init__(
        self,
        key_index: Sequence[str],
        values: Sequence[str],
        ignore_unique: bool = True,
        order: Optional[Sequence[str]] = None,
        key_agg: Optional[Sequence[str]] = None,
        agg_args: Sequence[Any] = ("sum",),
        agg_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.key_index = key_index
        self.values = values
        self.ignore_unique = ignore_unique
        self.order = order
        self.key_agg = key_agg
        self.agg_args = agg_args
        self.agg_kwargs = agg_kwargs

    def __repr__(self) -> str:
        "usual"
        return string_sig(self)  # type: ignore[arg-type]


class CubeLogs:
    """
    Processes logs coming from experiments.
    """

    def __init__(
        self,
        data: Any,
        time: str = "date",
        keys: Sequence[str] = ("version_.*", "model_.*"),
        values: Sequence[str] = ("time_.*", "disc_.*"),
        ignored: Sequence[str] = (),
        recent: bool = False,
        formulas: Optional[Dict[str, Callable[[pandas.DataFrame], pandas.Series]]] = None,
    ):
        self._data = data
        self._time = time
        self._keys = keys
        self._values = values
        self._ignored = ignored
        self.recent = recent
        self._formulas = formulas

    def load(self, verbose: int = 0):
        """Loads and preprocesses the data. Returns self."""
        if isinstance(self._data, pandas.DataFrame):
            if verbose:
                print(f"[CubeLogs.load] load from dataframe, shape={self._data.shape}")
            self.data = self._data
        elif isinstance(self._data, list) and all(isinstance(r, dict) for r in self._data):
            if verbose:
                print(f"[CubeLogs.load] load from list of dicts, n={len(self._data)}")
            self.data = pandas.DataFrame(self._data)
        elif isinstance(self._data, list) and all(
            isinstance(r, pandas.DataFrame) for r in self._data
        ):
            if verbose:
                print(f"[CubeLogs.load] load from list of DataFrame, n={len(self._data)}")
            self.data = pandas.concat(self._data, axis=0)
        elif isinstance(self._data, list):
            cubes = []
            for item in enumerate_csv_files(self._data, verbose=verbose):
                df = open_dataframe(item)
                cube = CubeLogs(
                    df,
                    time=self._time,
                    keys=self._keys,
                    values=self._values,
                    ignored=self._ignored,
                    recent=self.recent,
                )
                cube.load()
                cubes.append(cube.data)
            self.data = pandas.concat(cubes, axis=0)
        else:
            raise NotImplementedError(
                f"Not implemented with the provided data (type={type(self._data)})"
            )

        assert all(isinstance(c, str) for c in self.data.columns), (
            f"The class only supports string as column names "
            f"but found {[c for c in self.data.columns if not isinstance(c, str)]}"
        )
        if verbose:
            print(f"[CubeLogs.load] loaded with shape={self.data.shape}")

        self._initialize_columns()
        if verbose:
            print(f"[CubeLogs.load] time={self.time}")
            print(f"[CubeLogs.load] keys={self.keys}")
            print(f"[CubeLogs.load] values={self.values}")
            print(f"[CubeLogs.load] ignored={self.ignored}")
            print(f"[CubeLogs.load] ignored_values={self.ignored_values}")
            print(f"[CubeLogs.load] ignored_keys={self.ignored_keys}")
        assert not (
            set(self.keys) & set(self.values)
        ), f"Columns {set(self.keys) & set(self.values)} cannot be keys and values"
        assert not (
            set(self.keys) & set(self.ignored)
        ), f"Columns {set(self.keys) & set(self.ignored)} cannot be keys and ignored"
        assert not (
            set(self.values) & set(self.ignored)
        ), f"Columns {set(self.keys) & set(self.ignored)} cannot be values and ignored"
        assert (
            self.time not in self.keys
            and self.time not in self.values
            and self.time not in self.ignored
        ), f"Column {self.time!r} is also a key, a value or ignored"
        self._columns = [self.time, *self.keys, *self.values, *self.ignored]
        self.dropped = [c for c in self.data.columns if c not in set(self.columns)]
        self.data = self.data[self.columns]
        if verbose:
            print(f"[CubeLogs.load] dropped={self.dropped}")
            print(f"[CubeLogs.load] data.shape={self.data.shape}")

        self._preprocess()
        if self.recent and verbose:
            print(f"[CubeLogs.load] keep most recent data.shape={self.data.shape}")

        # Let's apply the formulas
        if self._formulas:
            cols = set(self.data.columns)
            for k, f in self._formulas.items():
                if k in cols:
                    if verbose:
                        print(f"[CubeLogs.load] skip formula {k!r}")
                else:
                    if verbose:
                        print(f"[CubeLogs.load] apply formula {k!r}")
                    self.data[k] = f(self.data)
        self.values_for_key = {k: set(self.data[k]) for k in self.keys}
        nans = [
            c for c in [self.time, *self.keys] if self.data[c].isna().astype(int).sum() > 0
        ]
        assert not nans, f"The following keys {nans} have nan values. This is not allowed."
        if verbose:
            print(f"[CubeLogs.load] convert column {self.time!r} into date")
        self.data[self.time] = pandas.to_datetime(self.data[self.time])
        if verbose:
            print(f"[CubeLogs.load] done, shape={self.shape}")
        return self

    @property
    def shape(self) -> Tuple[int, int]:
        "Returns the shape."
        assert hasattr(self, "data"), "Method load was not called"
        return self.data.shape

    @property
    def columns(self) -> Sequence[str]:
        "Returns the columns."
        assert hasattr(self, "data"), "Method load was not called"
        return self.data.columns

    def _preprocess(self):
        last = self.values[0]
        gr = self.data[[self.time, *self.keys, last]].groupby([self.time, *self.keys]).count()
        gr = gr[gr[last] > 1]
        if self.recent:
            cp = self.data.copy()
            assert (
                "__index__" not in cp.columns
            ), f"'__index__' should not be a column in {cp.columns}"
            cp["__index__"] = np.arange(cp.shape[0])
            gr = (
                cp[[*self.keys, self.time, "__index__"]]
                .groupby(self.keys, as_index=False)
                .max()
            )
            filtered = pandas.merge(cp, gr, on=[self.time, "__index__", *self.keys])
            assert filtered.shape[0] <= self.data.shape[0], (
                f"Keeping the latest row brings more row {filtered.shape} "
                f"(initial is {self.data.shape})."
            )
            self.data = filtered.drop("__index__", axis=1)
        else:
            assert gr.shape[0] == 0, f"There are duplicated rows:\n{gr}"
            gr = self.data[[*self.keys, self.time]].groupby(self.keys).count()
            gr = gr[gr[self.time] > 1]
            assert (
                gr.shape[0] == 0
            ), f"recent should be true to keep the most recent row:\n{gr}"

    @classmethod
    def _filter_column(cls, filters, columns, can_be_empty=False):
        set_cols = set()
        for f in filters:
            reg = re.compile(f)
            cols = [c for c in columns if reg.search(c)]
            set_cols |= set(cols)
        assert (
            can_be_empty or set_cols
        ), f"Filters {filters} returns an empty set from {columns}"
        return sorted(set_cols)

    def _initialize_columns(self):
        self.keys = self._filter_column(self._keys, self.data.columns)
        self.values = self._filter_column(self._values, self.data.columns)
        self.ignored = self._filter_column(self._ignored, self.data.columns, True)
        assert (
            self._time in self.data.columns
        ), f"Column {self._time} not found in {self.data.columns}"
        ignored_keys = set(self.ignored) & set(self.keys)
        ignored_values = set(self.ignored) & set(self.values)
        self.keys = [c for c in self.keys if c not in ignored_keys]
        self.values = [c for c in self.values if c not in ignored_values]
        self.ignored_keys = sorted(ignored_keys)
        self.ignored_values = sorted(ignored_values)
        self.time = self._time

    def __str__(self) -> str:
        "usual"
        return str(self.data) if hasattr(self, "data") else str(self._data)

    def view(self, view_def: CubeViewDef) -> pandas.DataFrame:
        """
        Returns a dataframe, a pivot view.
        `key_index` determines the index, the other key columns determines
        the columns. If `ignore_unique` is True, every columns with a unique value
        is removed.

        :param view_def: view definition
        :return: dataframe
        """
        key_agg = self._filter_column(view_def.key_agg, self.keys) if view_def.key_agg else []
        set_key_agg = set(key_agg)
        assert set_key_agg <= set(
            self.keys
        ), f"Non existing keys in key_agg {set_key_agg - set(self.keys)}"

        values = self._filter_column(view_def.values, self.values)
        assert set(values) <= set(
            self.values
        ), f"Non existing columns in values {set(values) - set(self.values)}"

        if key_agg:
            key_index = [
                c
                for c in self._filter_column(view_def.key_index, self.keys)
                if c not in set_key_agg
            ]
            keys_no_agg = [c for c in self.keys if c not in set_key_agg]
            data = (
                self.data[[*keys_no_agg, *values]]
                .groupby(key_index, as_index=False)
                .agg(*view_def.agg_args, **(view_def.agg_kwargs or {}))
            )
        else:
            key_index = self._filter_column(view_def.key_index, self.keys)
            data = self.data[[*self.keys, *values]]

        assert set(key_index) <= set(
            self.keys
        ), f"Non existing keys in key_index {set(key_index) - set(self.keys)}"

        set_key_columns = {
            c for c in self.keys if c not in key_index and c not in set(key_agg)
        }
        if view_def.ignore_unique:
            key_index = [k for k in key_index if len(self.values_for_key[k]) > 1]
            key_columns = [k for k in set_key_columns if len(self.values_for_key[k]) > 1]
        else:
            key_columns = sorted(set_key_columns)

        if view_def.order:
            assert set(view_def.order) <= set_key_columns, (
                f"Non existing columns from order in key_columns "
                f"{set(view_def.order) - set_key_columns}"
            )
            key_columns = [
                *view_def.order,
                *[c for c in key_columns if c not in view_def.order],
            ]
        return data.pivot(index=key_index[::-1], columns=key_columns, values=values)

    def describe(self) -> pandas.DataFrame:
        """Basic description of all variables."""
        rows = []
        for name in self.data.columns:
            values = self.data[name]
            dtype = values.dtype
            nonan = values.dropna()
            obs = dict(
                name=name,
                dtype=str(dtype),
                missing=len(values) - len(nonan),
            )
            if len(nonan) > 0:
                obs.update(
                    dict(
                        min=nonan.min(),
                        max=nonan.max(),
                        count=len(nonan),
                    )
                )
                if is_numeric_dtype(nonan):
                    obs.update(
                        dict(
                            mean=nonan.mean(),
                            sum=nonan.sum(),
                        )
                    )
                else:
                    unique = set(nonan)
                    obs["n_values"] = len(unique)
                    if len(unique) < 20:
                        obs["values"] = ",".join(map(str, sorted(unique)))
            rows.append(obs)
        return pandas.DataFrame(rows).set_index("name")

    def to_excel(
        self,
        output: str,
        views: Dict[str, CubeViewDef],
        main: Optional[str] = "main",
        raw: Optional[str] = "raw",
        verbose: int = 0,
    ):
        """
        Creates an excel file with a list of view.

        :param output: output file to create
        :param views: list of views to append
        :param main: add a page with statitcs on all variables
        :param raw: add a page with the raw data
        :param verbose: verbosity
        """

        with pandas.ExcelWriter(output, engine="openpyxl") as writer:
            if main:
                assert main not in views, f"{main!r} is duplicated in views {sorted(views)}"
                df = self.describe()
                if verbose:
                    print(f"[CubeLogs.to_helper] add sheet {main!r} with shape {df.shape}")
                df.to_excel(writer, sheet_name=main, freeze_panes=(1, 1))
                self._apply_excel_style(main, writer, df)
            if raw:
                assert main not in views, f"{main!r} is duplicated in views {sorted(views)}"
                if verbose:
                    print(f"[CubeLogs.to_helper] add sheet {raw!r} with shape {self.shape}")
                self.data.to_excel(writer, sheet_name=raw, freeze_panes=(1, 1), index=True)
                self._apply_excel_style(raw, writer, self.data)

            for name, view in views.items():
                df = self.view(view)
                if verbose:
                    print(
                        f"[CubeLogs.to_helper] add sheet {name!r} with shape "
                        f"{df.shape}, index={df.index.names}, columns={df.columns.names}"
                    )
                df.to_excel(
                    writer,
                    sheet_name=name,
                    freeze_panes=(df.index.nlevels, df.columns.nlevels),
                )
                self._apply_excel_style(name, writer, df)
        if verbose:
            print(f"[CubeLogs.to_helper] done with {len(views)} views")

    def _apply_excel_style(self, name: str, writer: pandas.ExcelWriter, df: pandas.DataFrame):
        from openpyxl.styles import Alignment
        from openpyxl.utils import get_column_letter

        # from openpyxl.styles import Font, PatternFill, numbers

        left = Alignment(horizontal="left")
        right = Alignment(horizontal="right")
        # center = Alignment(horizontal="center")
        # bold_font = Font(bold=True)
        # red = Font(color="FF0000")
        # yellow = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
        # redf = PatternFill(start_color="FF0000", end_color="FF0000", fill_type="solid")

        sheet = writer.sheets[name]
        n_rows = df.shape[0] + df.columns.nlevels + df.index.nlevels
        n_cols = df.shape[1] + df.index.nlevels
        co: Dict[int, int] = {}
        sizes: Dict[int, int] = {}
        cols = set()
        for i in range(1, n_rows):
            for j, cell in enumerate(sheet[i]):
                if j > n_cols:
                    break
                cols.add(cell.column)
                if isinstance(cell.value, float):
                    co[j] = co.get(j, 0) + 1
                elif isinstance(cell.value, str):
                    sizes[cell.column] = max(sizes.get(cell.column, 0), len(cell.value))

        for k, v in sizes.items():
            c = get_column_letter(k)
            sheet.column_dimensions[c].width = max(15, v)
        for k in cols:
            if k not in sizes:
                c = get_column_letter(k)
                sheet.column_dimensions[c].width = 15

        for i in range(1, n_rows):
            for j, cell in enumerate(sheet[i]):
                if j > n_cols:
                    break
                if isinstance(cell.value, pandas.Timestamp):
                    cell.alignment = right
                    dt = cell.value.to_pydatetime()
                    cell.value = dt
                    cell.number_format = (
                        "YYYY-MM-DD"
                        if (
                            dt.hour == 0
                            and dt.minute == 0
                            and dt.second == 0
                            and dt.microsecond == 0
                        )
                        else "YYYY-MM-DD 00:00:00"
                    )
                elif isinstance(cell.value, (float, int)):
                    cell.alignment = right
                    x = abs(cell.value)
                    if int(x) == x:
                        cell.number_format = "0"
                    elif x > 5000:
                        cell.number_format = "# ##0"
                    elif x >= 500:
                        cell.number_format = "0.0"
                    elif x >= 50:
                        cell.number_format = "0.00"
                    elif x >= 5:
                        cell.number_format = "0.000"
                    elif x > 0.5:
                        cell.number_format = "0.0000"
                    elif x > 0.005:
                        cell.number_format = "0.00000"
                    else:
                        cell.number_format = "0.000E+00"
                else:
                    cell.alignment = left
