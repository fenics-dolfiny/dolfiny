from ufl.algebra import Division, Product, Sum
from ufl.argument import Argument
from ufl.coefficient import Coefficient
from ufl.constant import Constant
from ufl.constantvalue import FloatValue, IntValue
from ufl.core.expr import Expr
from ufl.core.multiindex import MultiIndex
from ufl.corealg.traversal import unique_post_traversal
from ufl.form import Form
from ufl.indexed import Indexed
from ufl.indexsum import IndexSum
from ufl.measure import integral_type_to_measure_name
from ufl.referencevalue import ReferenceValue
from ufl.tensors import ComponentTensor
from ufl.variable import Label

from dolfiny.units import Quantity


def visualize(expr, filename, label=""):
    try:
        import pygraphviz as pgv
    except ImportError:
        raise ImportError("pygraphviz not found.")

    # strict=False to show all edges for e.g. product of a node with itself
    G = pgv.AGraph(directed=True, strict=True)

    if isinstance(expr, Form):
        G.add_node(expr, label="Form")

        for i, integral in enumerate(expr.integrals()):
            integrand = integral.integrand()
            add_expression_graph(integrand, G, f"Integrand {i}")
            itg_type = integral.integral_type()
            measure_name = integral_type_to_measure_name[itg_type]
            measure_label = f"{measure_name}({integral.ufl_domain()})"

            G.add_edge(expr, integrand, label=measure_label)
    elif isinstance(expr, Expr):
        add_expression_graph(expr, G, label)
    else:
        raise RuntimeError(f"Cannot visualize graph of {expr.__class__.__name__}.")

    G.layout("dot")
    G.draw(filename)


def add_expression_graph(expr, G, name):
    _skip_nodes = (MultiIndex, Label)

    nodes = []
    for n in unique_post_traversal(expr):
        nodes.append(n)

        if isinstance(n, Sum):
            label = "+"
        elif isinstance(n, Product):
            label = "*"
        elif isinstance(n, Division):
            label = "/"
        elif isinstance(n, IntValue | FloatValue):
            label = n.value()
        elif isinstance(n, ReferenceValue | Argument | Constant | Coefficient):
            label = str(n)
        elif isinstance(n, IndexSum):
            label = f"Σ {n.index()}"
        elif isinstance(n, ComponentTensor):
            label = f"ComponentTensor({n.indices()})"
        elif isinstance(n, Indexed):
            label = f"Indexed({n.ufl_operands[1]})"
        elif isinstance(n, _skip_nodes):
            continue
        else:
            label = n.__class__.__name__

        try:
            ufl_shape = getattr(n, "ufl_shape")
            shape = "box"
            xlabel = f"{ufl_shape}"
        except (AttributeError, ValueError):
            shape = "ellipse"
            xlabel = ""

        if isinstance(n, Quantity):
            color = "blue"
            fontcolor = color
            label = f"{label}"
        else:
            color = "black"
            fontcolor = color

        G.add_node(n, label=label, xlabel=xlabel, shape=shape, color=color, fontcolor=fontcolor)
        for i, e in enumerate(n.ufl_operands):
            if isinstance(e, _skip_nodes):
                continue
            G.add_edge(n, e, arrowhead="normal", xlabel=f"{i}", labelfontsize=10)

    # Group all nodes into a subgraph and add a box around
    G.add_subgraph(nodes, name=f"cluster {name}", label=name, cluster=True, newrank=True)

    return G
