"""Turn a .bench netlist into multi-level Boolean expressions for each target bit."""
import re

import pandas as pd

GATES = {'NAND', 'AND', 'OR', 'NOT', 'DFF', 'NOR'}


def parse_netlist(netlist):
    """Map each signal to its first-level expression, e.g. {'U203': 'NAND(U204,U205)'}.

    A DFF output maps to the name of its D input.
    """
    boolean_expressions = {}
    for line in netlist.splitlines():
        # Ignore comments and empty lines
        if line.startswith('#') or not line.strip():
            continue
        if "AND(" in line or "NAND(" in line or "OR(" in line or "NOT(" in line or "NOR(" in line:
            output, input_ = line.replace(" ", "").split('=')
            boolean_expressions[output] = input_
            continue
        if "DFF" in line:
            output, input_ = line.replace(" ", "").split('=')
            input_ = input_.replace("DFF(", "").replace(")", "")
            boolean_expressions[output] = input_
            continue
    return boolean_expressions


def generate_next_level_expressions(cur_expressions, first_boolean_expressions):
    """Expand every input signal of each expression by one more level of logic."""
    next_level_expressions = {}
    for component, expression in cur_expressions.items():
        tmp = expression
        # Skip the first token (the outermost gate name)
        variables = list(set(re.findall(r'\b\w+\b', expression)[1:]))
        # Substitute through lowercase placeholders so a replacement is never expanded twice
        count = 0
        replace_map = {}
        for var in variables:
            if var in first_boolean_expressions:
                placeholder = chr(ord('a') + count) * (count + 1)
                count += 1
                tmp = tmp.replace(var, placeholder)
                replace_map[placeholder] = f"{first_boolean_expressions[var]}"
        for placeholder, replacement in replace_map.items():
            tmp = tmp.replace(placeholder, replacement)
        next_level_expressions[component] = tmp
    return next_level_expressions


def generate_select_level_expressions(first_boolean_expressions, level):
    cur_expressions = first_boolean_expressions
    for _ in range(level - 1):
        cur_expressions = generate_next_level_expressions(cur_expressions, first_boolean_expressions)
    return cur_expressions


def read_words(word_path):
    """Read a .words file: one word per line, bits separated by whitespace."""
    with open(word_path, 'r') as file:
        return [line.split() for line in file if line.strip()]


def filter_target_bits(expressions, word_path):
    items_list = [item for word in read_words(word_path) for item in word]
    return {key: expressions[key] for key in items_list}


def replace_variables_with_indices(expression):
    """Anonymize signal names: 'NAND(U1,U2)' -> 'NAND(X[0],X[1])'."""
    variables = re.findall(r'\b\w+\b', expression)
    unique_variables = [var for var in dict.fromkeys(variables) if var not in GATES]
    variable_to_index = {var: f"X[{i}]" for i, var in enumerate(unique_variables)}
    for var, replacement in variable_to_index.items():
        expression = re.sub(rf'\b{var}\b', replacement, expression)
    return expression


def netlist2expressions(netlist, level, word_path):
    """Return a DataFrame indexed by target bit with its `level`-level expression (raw and anonymized)."""
    first_boolean_expressions = parse_netlist(netlist)
    cur_expressions = generate_select_level_expressions(first_boolean_expressions, level)
    cur_expressions = filter_target_bits(cur_expressions, word_path)
    expressions_df = pd.DataFrame([cur_expressions]).T
    expressions_df.columns = [f'{level}_level_expr']
    expressions_df[f'{level}_level_expr_x'] = expressions_df[f'{level}_level_expr'].apply(replace_variables_with_indices)
    expressions_df.sort_index(inplace=True)
    return expressions_df
