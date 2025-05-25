from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
import onnx
import onnx.helper as oh


class GraphRendering:
    """
    Helpers to renders a graph.

    :param proto: model or graph to render.
    """

    def __init__(self, proto: Union[onnx.FunctionProto, onnx.GraphProto, onnx.ModelProto]):
        self.proto = proto

    def __repr__(self) -> str:
        "usual"
        return f"{self.__class__.__name__}(<{self.proto.__class__.__name__}>)"

    @classmethod
    def computation_order(
        cls,
        nodes: Sequence[onnx.NodeProto],
        existing: Optional[List[str]] = None,
        start: int = 1,
    ) -> List[int]:
        """
        Returns the soonest a node can be computed,
        every node can assume all nodes with a lower number exists.
        Every node with a higher number must wait for the previous one.

        :param nodes: list of nodes
        :param existing: existing before any computation starts
        :param start: lower number
        :return: computation order
        """
        assert not ({"If", "Scan", "Loop", "SequenceMap"} & set(n.op_type for n in nodes)), (
            f"This algorithme is not yet implemented if the sequence contains "
            f"a control flow, types={sorted(set(n.op_type for n in nodes))}"
        )
        number = {e: start - 1 for e in (existing or [])}  # noqa: C420
        results = [start for _ in nodes]
        for i_node, node in enumerate(nodes):
            assert all(i in number for i in node.input), (
                f"Missing input in node {i_node} type={node.op_type}: "
                f"{[i for i in node.input if i not in number]}"
            )
            mx = max(number[i] for i in node.input) + 1
            results[i_node] = mx
            for i in node.output:
                number[i] = mx
        return results

    @classmethod
    def graph_positions(
        cls,
        nodes: Sequence[onnx.NodeProto],
        order: List[int],
        existing: Optional[List[str]] = None,
    ) -> List[Tuple[int, int]]:
        """
        Returns positions on a plan for every node in a graph.
        The function minimizes the number of lines crossing each others.
        It goes forward, every line is optimized depending on what is below.
        It could be improved with more iterations.

        :param nodes: list of nodes
        :param existing: existing names
        :param order: computation order returned by
            :meth:`onnx_diagnostic.helpers.graph_helper.GraphRendering.computation_order`
        :return: list of tuple( row, column)
        """
        # initialization
        min_row = min(order)
        n_rows = max(order) + 1
        names: Dict[str, int] = {}

        positions = [(min_row, i) for i in range(len(order))]
        for row in range(min_row, n_rows):
            indices = [i for i, o in enumerate(order) if o == row]
            assert indices, f"indices cannot be empty for row={row}, order={order}"
            ns = [nodes[i] for i in indices]
            mx = [max(names.get(i, 0) for i in n.input) for n in ns]
            mix = [(m, i) for i, m in enumerate(mx)]
            mix.sort()
            for c, (_m, i) in enumerate(mix):
                positions[indices[i]] = (row, c)
                n = nodes[indices[i]]
                for o in n.output:
                    names[o] = c

        return positions

    @classmethod
    def text_positions(
        cls, nodes: Sequence[onnx.NodeProto], positions: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        Returns positions for the nodes assuming it is rendered into text.

        :param nodes: list of nodes
        :param positions: positions returned by
            :meth:`onnx_diagnostic.helpers.graph_helper.GraphRendering.graph_positions`
        :return: text positions
        """
        new_positions = [(row * 4, col * 2 + row) for row, col in positions]
        column_size = {col: 3 for _, col in new_positions}
        for i, (_row, col) in enumerate(new_positions):
            size = len(nodes[i].op_type) + 5
            column_size[col] = max(column_size[col], size)

        # cumulated
        sort = sorted(column_size.items())
        cumul = dict(sort[:1])
        results = {sort[0][0]: sort[0][1] // 2}
        for col, size in sort[1:]:
            c = sum(cumul.values())
            cumul[col] = c
            results[col] = c + size // 2
        return [(row, results[col]) for row, col in new_positions]

    @property
    def nodes(self) -> List[onnx.NodeProto]:
        "Returns the list of nodes"
        return (
            self.proto.graph.node
            if isinstance(self.proto, onnx.ModelProto)
            else self.proto.node
        )

    @property
    def start_names(self) -> List[onnx.NodeProto]:
        "Returns the list of known names, inputs and initializer"
        graph = self.proto.graph if isinstance(self.proto, onnx.ModelProto) else self.proto
        input_names = (
            list(graph.input)
            if isinstance(graph, onnx.FunctionProto)
            else [i.name for i in graph.input]
        )
        init_names = (
            []
            if isinstance(graph, onnx.FunctionProto)
            else [
                *[i.name for i in graph.initializer],
                *[i.name for i in graph.sparse_initializer],
            ]
        )
        return [*input_names, *init_names]

    @property
    def input_names(self) -> List[str]:
        "Returns the list of input names."
        return (
            self.proto.input
            if isinstance(self.proto, onnx.FunctionProto)
            else [
                i.name
                for i in (
                    self.proto if isinstance(self.proto, onnx.GraphProto) else self.proto.graph
                ).input
            ]
        )

    @property
    def output_names(self) -> List[str]:
        "Returns the list of output names."
        return (
            self.proto.output
            if isinstance(self.proto, onnx.FunctionProto)
            else [
                i.name
                for i in (
                    self.proto if isinstance(self.proto, onnx.GraphProto) else self.proto.graph
                ).output
            ]
        )

    @classmethod
    def build_node_edges(cls, nodes: Sequence[onnx.NodeProto]) -> Set[Tuple[int, int]]:
        """Builds the list of edges between nodes."""
        produced = {}
        for i, node in enumerate(nodes):
            for o in node.output:
                produced[o] = i
        edges = set()
        for i, node in enumerate(nodes):
            for name in node.input:
                if name in produced:
                    edge = produced[name], i
                    edges.add(edge)
        return edges

    @classmethod
    def text_grid(cls, grid: List[List[str]], position: Tuple[int, int], text: str):
        """
        Prints inplace a text in a grid. The text is centered.

        :param grid: grid
        :param position: position
        :param text: text to print
        """
        row, col = position
        begin = col - len(text) // 2
        grid[row][begin : begin + len(text)] = list(text)

    def text_edge(
        cls,
        grid: List[List[str]],
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        mode: str = "square",
    ):
        """
        Prints inplace an edge in a grid. The text is centered.

        :param grid: grid
        :param p1: first position
        :param p2: second position
        :param mode: ``'square'`` is the only supported value
        """
        assert mode == "square", f"mode={mode!r} not supported"
        assert p1[0] < p2[0], f"Unexpected edge p1={p1}, p2={p2}"
        assert p1[0] + 2 <= p2[0] - 2, f"Unexpected edge p1={p1}, p2={p2}"
        # removes this when the algorithm is ready
        assert 0 <= p1[0] < len(grid) - 3, f"p1={p1}, grid:{len(grid)},{len(grid[0])}"
        assert 2 <= p2[0] < len(grid) - 1, f"p2={p2}, grid:{len(grid)},{len(grid[0])}"
        assert (
            0 <= p1[1] < min(len(g) for g in grid)
        ), f"p1={p1}, sizes={[len(g) for g in grid]}"
        assert (
            0 <= p2[1] < min(len(g) for g in grid)
        ), f"p2={p2}, sizes={[len(g) for g in grid]}"
        grid[p1[0] + 1][p1[1]] = "|"
        grid[p1[0] + 2][p1[1]] = "+"

        if p1[0] + 2 == p2[0] - 2:
            a, b = (p1[1] + 1, p2[1] - 1) if p1[1] < p2[1] else (p2[1] + 1, p1[1] - 1)
            grid[p1[0] + 2][a : b + 1] = ["-"] * (b - a + 1)
        else:
            middle = (p1[1] + p2[1]) // 2
            a, b = (p1[1] + 1, middle - 1) if p1[1] < middle else (middle + 1, p1[1] - 1)
            grid[p1[0] + 2][a : b + 1] = ["-"] * (b - a + 1)
            a, b = (p1[1] + 1, middle - 1) if p1[1] < middle else (middle + 1, p1[1] - 1)
            grid[p1[0] + 2][a : b + 1] = ["-"] * (b - a + 1)

            grid[p1[0] + 2][middle] = "+"
            grid[p2[0] - 2][middle] = "+"

        grid[p2[0] - 2][p2[1]] = "+"
        grid[p2[0] - 1][p2[1]] = "|"

    def text_rendering(self, prefix="") -> str:
        """
        Renders a model in text.

        .. runpython::
            :showcode:

            import textwrap
            import onnx
            import onnx.helper as oh
            from onnx_diagnostic.helpers.graph_helper import GraphRendering

            TFLOAT = onnx.TensorProto.FLOAT

            proto = oh.make_model(
                oh.make_graph(
                    [
                        oh.make_node("Add", ["X", "Y"], ["xy"]),
                        oh.make_node("Neg", ["Y"], ["ny"]),
                        oh.make_node("Mul", ["xy", "ny"], ["a"]),
                        oh.make_node("Mul", ["a", "Y"], ["Z"]),
                    ],
                    "-nd-",
                    [
                        oh.make_tensor_value_info("X", TFLOAT, ["a", "b", "c"]),
                        oh.make_tensor_value_info("Y", TFLOAT, ["a", "b", "c"]),
                    ],
                    [oh.make_tensor_value_info("Z", TFLOAT, ["a", "b", "c"])],
                ),
                opset_imports=[oh.make_opsetid("", 18)],
                ir_version=9,
            )
            graph = GraphRendering(proto)
            text = textwrap.dedent(graph.text_rendering()).strip("\\n")
            print(text)
        """
        nodes = [
            *[oh.make_node(i, ["BEGIN"], [i]) for i in self.input_names],
            *self.nodes,
            *[oh.make_node(i, [i], ["END"]) for i in self.output_names],
        ]
        existing = set(self.start_names) - set(self.input_names)
        existing |= {"BEGIN"}
        existing = sorted(existing)
        order = self.computation_order(nodes, existing)
        positions = self.graph_positions(nodes, order, existing)
        text_pos = self.text_positions(nodes, positions)
        edges = self.build_node_edges(nodes)
        max_len = max(col for _, col in text_pos) + max(len(n.op_type) for n in nodes)
        max_row = max(row for row, _ in text_pos) + 2
        grid = [[" " for i in range(max_len + 1)] for _ in range(max_row + 1)]

        for n1, n2 in edges:
            self.text_edge(grid, text_pos[n1], text_pos[n2])
            assert len(set(len(g) for g in grid)) == 1, f"lengths={[len(g) for g in grid]}"
        for node, pos in zip(nodes, text_pos):
            self.text_grid(grid, pos, node.op_type)
            assert len(set(len(g) for g in grid)) == 1, f"lengths={[len(g) for g in grid]}"

        return "\n".join(
            f"{prefix}{line.rstrip()}" for line in ["".join(line) for line in grid]
        )
