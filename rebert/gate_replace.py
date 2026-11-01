"""Netlist rewriting used to corrupt structural patterns.

* convert2ANDINVERTER: rewrite an AND/NAND/OR/NOR gate as 2-input ANDs and inverters.
* convert2TwoInputs:   split a wide gate into a tree of 2-input gates of the same family.
* replace_gates:       apply AND-inverter rewriting to a fraction of the gates
                       (the "gate replacement index"), then split the rest into 2-input gates.
"""
import random

GATE_PREFIXES = (' = AND(', ' = NAND(', ' = NOR(', ' = OR(')


def get_unique_signal(base_name, used_signals):
    index = 1
    new_name = f"{base_name}_{index}"
    while new_name in used_signals:
        index += 1
        new_name = f"{base_name}_{index}"
    used_signals.add(new_name)
    return new_name

def convert2ANDINVERTER(line, used_signals):
    
    # Apply transformation to AND gate
    if ' = AND(' in line:
        parts = line.split(' = AND(')
        inputs = parts[1].strip(')\n').split(', ')
        if len(inputs) == 3:
            A, B, C, D = parts[0], inputs[0], inputs[1], inputs[2]
            A_1 = get_unique_signal(A, used_signals)
            return [f"{A_1} = AND({B}, {C})", f"{A} = AND({A_1}, {D})"]
        if len(inputs) == 4:
            A, B, C, D, E = parts[0], inputs[0], inputs[1], inputs[2], inputs[3]
            A_1 = get_unique_signal(A, used_signals)
            A_2 = get_unique_signal(A, used_signals)
            return [f"{A_1} = AND({B}, {C})", f"{A_2} = AND({D}, {E})", f"{A} = AND({A_1}, {A_2})"]
        if len(inputs) == 5:
            A, B, C, D, E, F = parts[0], inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
            A_1 = get_unique_signal(A, used_signals)
            A_2 = get_unique_signal(A, used_signals)
            A_3 = get_unique_signal(A, used_signals)
            return [f"{A_1} = AND({B}, {C})", f"{A_2} = AND({D}, {E})", f"{A_3} = AND({A_1}, {A_2})", f"{A} = AND({F}, {A_3})"]

    # Apply transformation to 3-input NAND gate
    elif ' = NAND(' in line:
        parts = line.split(' = NAND(')
        inputs = parts[1].strip(')\n').split(', ')
        if len(inputs) == 2:
            A, B, C = parts[0], inputs[0], inputs[1]
            A_1 = get_unique_signal(A, used_signals)
            return [f"{A_1} = AND({B}, {C})", f"{A} = NOT({A_1})"]
        if len(inputs) == 3:
            A, B, C, D = parts[0], inputs[0], inputs[1], inputs[2]
            A_1 = get_unique_signal(A, used_signals)
            A_2 = get_unique_signal(A, used_signals)
            return [f"{A_1} = AND({B}, {C})", f"{A_2} = AND({A_1}, {D})", f"{A} = NOT({A_2})"]
        if len(inputs) == 4:
            A, B, C, D, E = parts[0], inputs[0], inputs[1], inputs[2], inputs[3]
            A_1 = get_unique_signal(A, used_signals)
            A_2 = get_unique_signal(A, used_signals)
            A_3 = get_unique_signal(A, used_signals)
            return [f"{A_1} = AND({B}, {C})", f"{A_2} = AND({D}, {E})", f"{A_3} = AND({A_1}, {A_2})", f"{A} = NOT({A_3})"]
        if len(inputs) == 5:
            A, B, C, D, E, F = parts[0], inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
            A_1 = get_unique_signal(A, used_signals)
            A_2 = get_unique_signal(A, used_signals)
            A_3 = get_unique_signal(A, used_signals)
            A_4 = get_unique_signal(A, used_signals)
            return [f"{A_1} = AND({B}, {C})", f"{A_2} = AND({D}, {E})", f"{A_3} = AND({A_1}, {A_2})", f"{A_4} = AND({F}, {A_3})", f"{A} = NOT({A_4})"]
    
    # Apply transformation NOR gate
    elif ' = NOR(' in line:
        parts = line.split(' = NOR(')
        inputs = parts[1].strip(')\n').split(', ')
        if len(inputs) == 2:
            A, B, C = parts[0], inputs[0], inputs[1]
            A_1 = get_unique_signal(A, used_signals)
            A_2 = get_unique_signal(A, used_signals)
            return [f"{A_1} = NOT({B})", f"{A_2} = NOT({C})", f"{A} = AND({A_1}, {A_2})"]
        if len(inputs) == 3:
            A, B, C, D = parts[0], inputs[0], inputs[1], inputs[2]
            A_1 = get_unique_signal(A, used_signals)
            A_2 = get_unique_signal(A, used_signals)
            A_3 = get_unique_signal(A, used_signals)
            A_4 = get_unique_signal(A, used_signals)
            return [f"{A_1} = NOT({B})", f"{A_2} = NOT({C})", f"{A_3} = NOT({D})", f"{A_4} = AND({A_1}, {A_2})", f"{A} = AND({A_3}, {A_4})"]
        if len(inputs) == 4:
            A, B, C, D, E = parts[0], inputs[0], inputs[1], inputs[2], inputs[3]
            A_1 = get_unique_signal(A, used_signals)
            A_2 = get_unique_signal(A, used_signals)
            A_3 = get_unique_signal(A, used_signals) 
            A_4 = get_unique_signal(A, used_signals)
            A_5 = get_unique_signal(A, used_signals)
            A_6 = get_unique_signal(A, used_signals)
            return [f"{A_1} = NOT({B})", f"{A_2} = NOT({C})", f"{A_3} = NOT({D})", f"{A_4} = NOT({E})", f"{A_5} = AND({A_1}, {A_2})", f"{A_6} = AND({A_3}, {A_4})", f"{A} = AND({A_5}, {A_6})"]
        if len(inputs) == 5:
            A, B, C, D, E, F = parts[0], inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
            A_1 = get_unique_signal(A, used_signals)
            A_2 = get_unique_signal(A, used_signals)
            A_3 = get_unique_signal(A, used_signals)
            A_4 = get_unique_signal(A, used_signals)
            A_5 = get_unique_signal(A, used_signals)
            A_6 = get_unique_signal(A, used_signals)
            A_7 = get_unique_signal(A, used_signals)
            A_8 = get_unique_signal(A, used_signals)
            return [f"{A_1} = NOT({B})", f"{A_2} = NOT({C})", f"{A_3} = NOT({D})", f"{A_4} = NOT({E})", f"{A_5} = NOT({F})", f"{A_6} = AND({A_1}, {A_2})", 
                    f"{A_7} = AND({A_3}, {A_4})", f"{A_8} = AND({A_5}, {A_6})", f"{A} = AND({A_7}, {A_8})"]

    elif ' = OR(' in line:
        parts = line.split(' = OR(')
        inputs = parts[1].strip(')\n').split(', ')
        if len(inputs) == 2:
            A, B, C = parts[0], inputs[0], inputs[1]
            A_1 = get_unique_signal(A, used_signals)
            A_2 = get_unique_signal(A, used_signals)
            A_3 = get_unique_signal(A, used_signals)
            return [f"{A_1} = NOT({B})", f"{A_2} = NOT({C})", f"{A_3} = AND({A_1}, {A_2})", f"{A} = NOT({A_3})"]
        if len(inputs) == 3:
            A, B, C, D = parts[0], inputs[0], inputs[1], inputs[2]
            A_1 = get_unique_signal(A, used_signals)
            A_2 = get_unique_signal(A, used_signals)
            A_3 = get_unique_signal(A, used_signals)
            A_4 = get_unique_signal(A, used_signals)
            A_5 = get_unique_signal(A, used_signals)
            return [f"{A_1} = NOT({B})", f"{A_2} = NOT({C})", f"{A_3} =  NOT({D})", f"{A_4} = AND({A_1}, {A_2})", f"{A_5} = AND({A_3}, {A_4})", f"{A} = NOT({A_5})"]
        if len(inputs) == 4:
            A, B, C, D, E = parts[0], inputs[0], inputs[1], inputs[2], inputs[3]
            A_1 = get_unique_signal(A, used_signals)
            A_2 = get_unique_signal(A, used_signals)
            A_3 = get_unique_signal(A, used_signals) 
            A_4 = get_unique_signal(A, used_signals)
            A_5 = get_unique_signal(A, used_signals)
            A_6 = get_unique_signal(A, used_signals)
            A_7 = get_unique_signal(A, used_signals)
            return [f"{A_1} = NOT({B})", f"{A_2} = NOT({C})", f"{A_3} = NOT({D})", f"{A_4} = NOT({E})", f"{A_5} = AND({A_1}, {A_2})", f"{A_6} = AND({A_3}, {A_4})", 
                    f"{A_7} = AND({A_5}, {A_6})", f"{A} = NOT({A_7})"]
        if len(inputs) == 5:
            A, B, C, D, E, F = parts[0], inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
            A_1 = get_unique_signal(A, used_signals)
            A_2 = get_unique_signal(A, used_signals)
            A_3 = get_unique_signal(A, used_signals)
            A_4 = get_unique_signal(A, used_signals)
            A_5 = get_unique_signal(A, used_signals)
            A_6 = get_unique_signal(A, used_signals)
            A_7 = get_unique_signal(A, used_signals)
            A_8 = get_unique_signal(A, used_signals)
            A_9 = get_unique_signal(A, used_signals)
            return [f"{A_1} = NOT({B})", f"{A_2} = NOT({C})", f"{A_3} = NOT({D})", f"{A_4} = NOT({E})", f"{A_5} = NOT({F})", f"{A_6} = AND({A_1}, {A_2})", f"{A_7} = AND({A_3}, {A_4})", 
                    f"{A_8} = AND({A_5}, {A_6})", f"{A_9} = AND({A_7}, {A_8})", f"{A} = NOT({A_9})"]
        
    # If no transformation is applied, return the original line
    return [line.strip()]

def replace_spaces_with_commas(line):
    # print(line)
    if ' = ' in line:
        parts = line.split(' = ')
        if '(' in parts[1] and ')' in parts[1]:
            func_part = parts[1].strip()
            func_name, inputs = func_part.split('(')
            inputs = inputs.strip(')\n').split()
            new_inputs = ', '.join(inputs)
            return f"{parts[0]} = {func_name}({new_inputs})"
    return line

def is_gate_line(line):
    return any(prefix in line for prefix in GATE_PREFIXES)


def transform_lines(lines, converter, used_signals, selected=None):
    """Apply `converter` to gate lines (only to indices in `selected`, if given) and normalize formatting."""
    transformed_netlist = []
    for i, line in enumerate(lines):
        if is_gate_line(line) and (selected is None or i in selected):
            transformed_netlist.extend(converter(line, used_signals))
        else:
            transformed_netlist.append(line.strip())

    unique_transformed_netlist = list(dict.fromkeys(transformed_netlist))
    cleaned_netlist = [line.replace("'", "").replace(",", "") for line in unique_transformed_netlist]
    return [replace_spaces_with_commas(line) for line in cleaned_netlist]


def read_lines(path):
    with open(path, 'r') as file:
        return file.readlines()


def write_lines(lines, path):
    with open(path, 'w') as file:
        for line in lines:
            file.write(line + '\n')


def gate_replacement(data_path, save_path):
    """Rewrite every AND/NAND/OR/NOR gate as AND-inverter logic."""
    write_lines(transform_lines(read_lines(data_path), convert2ANDINVERTER, set()), save_path)

def convert2TwoInputs(line, used_signals):
    
    # Apply transformation to AND gate
    if ' = AND(' in line:
        parts = line.split(' = AND(')
        inputs = parts[1].strip(')\n').split(', ')
        if len(inputs) == 3:
            A, B, C, D = parts[0], inputs[0], inputs[1], inputs[2]
            A_1 = get_unique_signal(A, used_signals)
            return [f"{A_1} = AND({B}, {C})", f"{A} = AND({A_1}, {D})"]
        if len(inputs) == 4:
            A, B, C, D, E = parts[0], inputs[0], inputs[1], inputs[2], inputs[3]
            A_1 = get_unique_signal(A, used_signals)
            A_2 = get_unique_signal(A, used_signals)
            return [f"{A_1} = AND({B}, {C})", f"{A_2} = AND({D}, {E})", f"{A} = AND({A_1}, {A_2})"]
        if len(inputs) == 5:
            A, B, C, D, E, F = parts[0], inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
            A_1 = get_unique_signal(A, used_signals)
            A_2 = get_unique_signal(A, used_signals)
            A_3 = get_unique_signal(A, used_signals)
            return [f"{A_1} = AND({B}, {C})", f"{A_2} = AND({D}, {E})", f"{A_3} = AND({A_1}, {A_2})", f"{A} = AND({F}, {A_3})"]

    # Apply transformation to 3-input NAND gate
    elif ' = NAND(' in line:
        parts = line.split(' = NAND(')
        inputs = parts[1].strip(')\n').split(', ')
        # if len(inputs) == 2:
        #     A, B, C = parts[0], inputs[0], inputs[1]
        #     A_1 = get_unique_signal(A, used_signals)
        #     return [f"{A_1} = AND({B}, {C})", f"{A} = NOT({A_1})"]
        if len(inputs) == 3:
            A, B, C, D = parts[0], inputs[0], inputs[1], inputs[2]
            A_1 = get_unique_signal(A, used_signals)
            return [f"{A_1} = AND({B}, {C})", f"{A} = NAND({A_1}, {D})"]
        if len(inputs) == 4:
            A, B, C, D, E = parts[0], inputs[0], inputs[1], inputs[2], inputs[3]
            A_1 = get_unique_signal(A, used_signals)
            A_2 = get_unique_signal(A, used_signals)
            A_3 = get_unique_signal(A, used_signals)
            return [f"{A_1} = AND({B}, {C})", f"{A_2} = AND({D}, {E})", f"{A} = NAND({A_1}, {A_2})"]
        if len(inputs) == 5:
            A, B, C, D, E, F = parts[0], inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
            A_1 = get_unique_signal(A, used_signals)
            A_2 = get_unique_signal(A, used_signals)
            A_3 = get_unique_signal(A, used_signals)
            return [f"{A_1} = AND({B}, {C})", f"{A_2} = AND({D}, {E})", f"{A_3} = AND({A_1}, {A_2})", f"{A} = NAND({F}, {A_3})"]
    
    # Apply transformation NOR gate
    elif ' = NOR(' in line:
        parts = line.split(' = NOR(')
        inputs = parts[1].strip(')\n').split(', ')
        # if len(inputs) == 2:
        #     A, B, C = parts[0], inputs[0], inputs[1]
        #     A_1 = get_unique_signal(A, used_signals)
        #     A_2 = get_unique_signal(A, used_signals)
        #     return [f"{A_1} = NOT({B})", f"{A_2} = NOT({C})", f"{A} = AND({A_1}, {A_2})"]
        if len(inputs) == 3:
            A, B, C, D = parts[0], inputs[0], inputs[1], inputs[2]
            A_1 = get_unique_signal(A, used_signals)
            return [f"{A_1} = OR({B}, {C})", f"{A} = NOR({A_1}, {D})"]
        if len(inputs) == 4:
            A, B, C, D, E = parts[0], inputs[0], inputs[1], inputs[2], inputs[3]
            A_1 = get_unique_signal(A, used_signals)
            A_2 = get_unique_signal(A, used_signals)
            return [f"{A_1} = OR({B}, {C})", f"{A_2} = OR({D}, {E})", f"{A} = NOR({A_1}, {A_2})"]
        if len(inputs) == 5:
            A, B, C, D, E, F = parts[0], inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
            A_1 = get_unique_signal(A, used_signals)
            A_2 = get_unique_signal(A, used_signals)
            A_3 = get_unique_signal(A, used_signals)
            return [f"{A_1} = OR({B}, {C})", f"{A_2} = OR({D}, {E})", f"{A_3} = OR({A_2}, {F})", f"{A} = NOR({A_1}, {A_3})"]

    elif ' = OR(' in line:
        parts = line.split(' = OR(')
        inputs = parts[1].strip(')\n').split(', ')
        # if len(inputs) == 2:
        #     A, B, C = parts[0], inputs[0], inputs[1]
        #     A_1 = get_unique_signal(A, used_signals)
        #     A_2 = get_unique_signal(A, used_signals)
        #     A_3 = get_unique_signal(A, used_signals)
        #     return [f"{A_1} = NOT({B})", f"{A_2} = NOT({C})", f"{A_3} = AND({A_1}, {A_2})", f"{A} = NOT({A_3})"]
        if len(inputs) == 3:
            A, B, C, D = parts[0], inputs[0], inputs[1], inputs[2]
            A_1 = get_unique_signal(A, used_signals)
            return [f"{A_1} = OR({B}, {C})", f"{A} = OR({A_1}, {D})"]
        if len(inputs) == 4:
            A, B, C, D, E = parts[0], inputs[0], inputs[1], inputs[2], inputs[3]
            A_1 = get_unique_signal(A, used_signals)
            A_2 = get_unique_signal(A, used_signals)
            return [f"{A_1} = OR({B}, {C})", f"{A_2} = OR({D}, {E})", f"{A} = OR({A_1}, {A_2})"]
        if len(inputs) == 5:
            A, B, C, D, E, F = parts[0], inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
            A_1 = get_unique_signal(A, used_signals)
            A_2 = get_unique_signal(A, used_signals)
            A_3 = get_unique_signal(A, used_signals)
            return [f"{A_1} = OR({B}, {C})", f"{A_2} = OR({D}, {E})", f"{A_3} = OR({A_2}, {F})", f"{A} = OR({A_1}, {A_3})"]
        
    # If no transformation is applied, return the original line
    return [line.strip()]


def netlist2TwoInputs(data_path, save_path):
    """Split every wide gate into 2-input gates of the same family."""
    write_lines(transform_lines(read_lines(data_path), convert2TwoInputs, set()), save_path)


def replace_gates(data_path, save_path, ratio, seed=0):
    """Gate replacement at index `ratio` in [0, 1].

    A random `ratio` fraction of the AND/NAND/OR/NOR gates is rewritten as
    AND-inverter logic; then every remaining wide gate is split into 2-input
    gates. ratio=0 keeps the original gate types, ratio=1 rewrites all of them.
    """
    lines = read_lines(data_path)
    gate_idx = [i for i, line in enumerate(lines) if is_gate_line(line)]
    n_replace = int(round(ratio * len(gate_idx)))
    selected = set(random.Random(seed).sample(gate_idx, n_replace))

    # Share used names across both passes so new signals never collide
    used_signals = set()
    lines = transform_lines(lines, convert2ANDINVERTER, used_signals, selected)
    lines = transform_lines(lines, convert2TwoInputs, used_signals)
    write_lines(lines, save_path)
