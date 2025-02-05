import gc
import os

import binaryninja as bn
import binaryninja.highlevelil
import networkx as nx
import pickle
from pyvis.network import Network

'''
This package uses Binary Ninja to extract all the graph version,
To normalize the data between all architectures an intermediate representation will be required
The BinaryNinja Medium Level IL contains data information which allows for a data flow analysis
The BinaryNinja High Level IL contains easily accessible AST information and includes the MLIL information
As a result HLIL will be used as the Intermediate Representation

'''


def extract_ast(filename) -> nx.DiGraph():
    '''

    :param filename: filename of the binary to analyze for the ast
    :param scope: scope of the binary LexicalBlock or Statement
    :return: returns an ast of the file
    '''
    project = bn.load(filename)
    ast = nx.DiGraph()
    node_id = 0

    def add_node(id, label):
        if not ast.has_node(id):
            ast.add_node(id, label=label)

    def add_edge(from_id, to_id, label):
        if not ast.has_edge(from_id, to_id):
            ast.add_edge(from_id, to_id, label=label)

    total_nodes = 0

    def visitor(operand_name, inst, instr_type_name, parent):
        nonlocal node_id, total_nodes
        # print("~~~~~~~~~~~~~~~~~~~~~~~")
        # print(inst.operation.name)
        # print(inst.function.__repr__(), inst.__repr__(), parent.__repr__())
        # print(inst.core_instr)
        string = f"{inst.operation.name}"
        add_node(node_id, string)
        total_nodes += 1
        statement_to_id[inst] = node_id
        if parent is not None:
            add_edge(statement_to_id[parent], node_id, label=0)
        else:
            add_edge(function_node_id, node_id, label=0)
        node_id += 1

    statement_to_id = {}

    for function in project.functions:
        if function.hlil:
            function_il = function.hlil
            function_node_id = node_id
            add_node(node_id, 'FUNCTION')
            node_id += 1
            for block in function_il.basic_blocks:
                statement_to_id[block] = node_id
                node_id += 1
                for statement in block:
                    statement.visit(visitor)
            statement_to_id.clear()

    # print(total_nodes, len(ast.nodes))

    project.file.close()
    del project
    gc.collect()
    return ast


def extract_ast_ddg(filename) -> nx.DiGraph():
    '''

    :param filename: the file name of the binary
    :param scope: what scoping to use, Statement or LexicalBlock
    :return: returns a merged graph of ast, and ddg
    '''
    project = bn.load(filename)
    astddg = nx.DiGraph()
    node_id = 0

    def add_node(id, label):
        if not astddg.has_node(id):
            astddg.add_node(id, label=label)

    def add_edge(from_id, to_id, label):
        if not astddg.has_edge(from_id, to_id):
            astddg.add_edge(from_id, to_id, label=label)

    total_nodes = 0

    def visitor(operand_name, inst, instr_type_name, parent):
        nonlocal node_id, total_nodes
        string = f"{inst.operation.name}"
        add_node(node_id, string)
        total_nodes += 1
        statement_to_id[inst] = node_id
        if parent is not None:
            add_edge(statement_to_id[parent], node_id, label=0)
        else:
            add_edge(function_node_id, node_id, label=0)
        node_id += 1

    vars = {}
    ddg_edges_to_add = []
    statement_to_id = {}

    for function in project.functions:
        if function.hlil:
            function_il = function.hlil
            function_node_id = node_id
            add_node(node_id, 'FUNCTION')
            node_id += 1
            #block = function_il.basic_blocks[0]
            for block in function_il.basic_blocks:
                statement_to_id[block] = node_id
                node_id += 1
                for statement in block:
                    statement.visit(visitor)

            ## DDG ANALYSIS
            for block in function_il.basic_blocks:
                for statement in block:
                    # print(statement)
                    for vars_read in statement.vars_read:
                        if vars_read.name in vars and str(vars_read.name) in str(statement):
                            node = vars[vars_read.name]
                            ddg_edges_to_add.append([node, statement_to_id[statement]])
                        else:
                            # print(f"{statement} wants {vars_read.name}")
                            # print(f"{statement.operation.name}")
                            # print(f"Variable {vars_read.name} never defined")
                            if statement.operation.name == "HLIL_VAR_DECLARE":
                                vars[vars_read.name] = statement_to_id[statement]
                    for vars_writ in statement.vars_written:
                        # print(f"{statement} VARS WRITTEN >>>> {vars_writ.name}")
                        vars[vars_writ.name] = statement_to_id[statement]

            statement_to_id.clear()

    # print(ddg_edges_to_add)
    for item in ddg_edges_to_add:
        add_edge(item[0], item[1], label=1)

    # print(total_nodes, len(astddg.nodes))

    project.file.close()
    del project
    gc.collect()
    return astddg


def extract_cfg_ast(filename) -> nx.DiGraph():
    '''

    :param filename: the file name of the binary
    :param scope: what scoping to use, Statement or LexicalBlock
    :return: returns a merged graph of cfg, and ast
    '''

    project = bn.load(filename)
    cfgast = nx.DiGraph()
    node_id = 0

    def add_node(id, label):
        nonlocal node_id
        if not cfgast.has_node(id):
            cfgast.add_node(id, label=label)

    def add_edge(from_id, to_id, label):
        if not cfgast.has_edge(from_id, to_id):
            cfgast.add_edge(from_id, to_id, label=label)

    total_nodes = 0

    def visitor(operand_name, inst, instr_type_name, parent):
        nonlocal node_id, total_nodes, first_inst
        string = f"{inst.operation.name}"
        if inst not in statement_to_id.keys():
            add_node(node_id, string)
            total_nodes += 1
            statement_to_id[inst] = node_id
            node_id += 1
        if parent is not None:
            add_edge(statement_to_id[parent], statement_to_id[inst], label=0)
        elif not first_inst:
            # print(parent, inst)
            add_edge(function_node_id, statement_to_id[inst], label=0)
            first_inst = True

    cfg_edges = []
    vars = {}
    statement_to_id = {}
    first_inst = False
    for function in project.functions:
        if not function.hlil:
            continue

        function_il = function.hlil

        function_node_id = node_id
        add_node(node_id, f'FUNCTION')
        for block in function_il.basic_blocks:
            statement_to_id[block] = node_id
            node_id += 1
            previous_statement = None
            for statement in block:
                statement.visit(visitor)
                if previous_statement is not None:
                    cfg_edges.append([statement_to_id[previous_statement], statement_to_id[statement]])
                previous_statement = statement

        for incoming_edge in block.incoming_edges:
            cfg_edges.append(
                [statement_to_id[incoming_edge.source[-1]], statement_to_id[incoming_edge.target[0]]])

        statement_to_id.clear()
        first_inst = False

    for item in cfg_edges:
        add_edge(item[0], item[1], label=2)

    project.file.close()
    del project
    gc.collect()
    return cfgast


def extract_cfg_ddg_ast(filename) -> nx.DiGraph():
    """

    :param filename: the file name of the binary
    :param scope: what scoping you want to use, Statement or LexicalBlock
    :return: returns a merged cfg, ddg, ast
    """
    """
        This is the aggregation of a
            - Control Flow Graph (CFG)
            - Abstract Syntax Tree (AST)
            - Data Dependence Graph (DDG)
    """
    project = bn.load(filename)
    cfgddgast = nx.DiGraph()
    node_id = 0

    def add_node(id, label):
        nonlocal node_id
        if not cfgddgast.has_node(id):
            cfgddgast.add_node(id, label=label)

    def add_edge(from_id, to_id, label):
        if not cfgddgast.has_edge(from_id, to_id):
            cfgddgast.add_edge(from_id, to_id, label=label)

    total_nodes = 0

    def visitor(operand_name, inst, instr_type_name, parent):
        nonlocal node_id, total_nodes, first_inst
        string = f"{inst.operation.name}"
        if inst not in statement_to_id.keys():
            add_node(node_id, string)
            total_nodes += 1
            statement_to_id[inst] = node_id
            node_id += 1
        if parent is not None:
            add_edge(statement_to_id[parent], statement_to_id[inst], label=0)
        elif not first_inst:
            # print(parent, inst)
            add_edge(function_node_id, statement_to_id[inst], label=0)
            first_inst = True

    cfg_edges = []
    vars = {}
    ddg_edges_to_add = []
    statement_to_id = {}
    first_inst = False
    for function in project.functions:
        if not function.hlil:
            continue

        function_il = function.hlil

        function_node_id = node_id
        add_node(node_id, f'FUNCTION')
        for block in function_il.basic_blocks:
            statement_to_id[block] = node_id
            node_id += 1
            previous_statement = None
            for statement in block:
                statement.visit(visitor)
                if previous_statement is not None:
                    cfg_edges.append([statement_to_id[previous_statement], statement_to_id[statement]])
                previous_statement = statement

        for incoming_edge in block.incoming_edges:
            cfg_edges.append(
                [statement_to_id[incoming_edge.source[-1]], statement_to_id[incoming_edge.target[0]]])

        ## DDG ANALYSIS
        for block in function_il.basic_blocks:
            for statement in block:
                for vars_read in statement.vars_read:
                    if vars_read.name in vars and str(vars_read.name) in str(statement):
                        node = vars[vars_read.name]
                        ddg_edges_to_add.append([node, statement_to_id[statement]])
                    else:
                        if statement.operation.name == "HLIL_VAR_DECLARE":
                            vars[vars_read.name] = statement_to_id[statement]
                for vars_writ in statement.vars_written:
                    vars[vars_writ.name] = statement_to_id[statement]

        statement_to_id.clear()
        first_inst = False

    for item in ddg_edges_to_add:
        add_edge(item[0], item[1], label=1)
    for item in cfg_edges:
        add_edge(item[0], item[1], label=2)

    project.file.close()
    del project
    gc.collect()
    return cfgddgast


def save_graph_to_file(graph, save_directory, file_name):
    """

    :param graph: graph object extracted from the above functions, should be a networkx DiGraph
    :param save_directory: file directory to save the graph object to
    :param file_name: file name for the saved graph
    :return:
    """
    with open('{}/{}'.format(save_directory, file_name), 'wb') as f:
        pickle.dump(graph, f)


def visualize_graph_html(graph, filename):
    """

    :param graph: graph object extracted from the above functions, should be a networkx DiGraph
    :param filename: should contain .html at the end, but if not specified it will be added
    :return:
    """
    if not filename.__contains__(".html"):
        filename += ".html"
    net = Network(notebook=True)
    net.from_nx(graph)
    for node in net.nodes:
        node['title'] = node['label']
    for edge in net.edges:
        edge['title'] = edge['label']

    net.show(filename)


def extract_graph_from_directory(directory, save_directory, graph_type):
    files_processed = 0
    graph_types = ['AST', 'CFGAST', 'ASTDDG', 'CFGDDGAST', 'ALL']
    if graph_type in graph_types:
        new_save_directory = '{}/{}'.format(save_directory, graph_type)
        os.makedirs(new_save_directory, exist_ok=True)
        os.makedirs(f'{new_save_directory}/AST/', exist_ok=True)
        os.makedirs(f'{new_save_directory}/CFGAST/', exist_ok=True)
        os.makedirs(f'{new_save_directory}/CFGDDGAST/', exist_ok=True)
        os.makedirs(f'{new_save_directory}/ASTDDG/', exist_ok=True)
        for foldername, subfolders, filenames in os.walk(directory):
            if graph_type != 'ALL':
                os.makedirs('{}/{}'.format(new_save_directory, foldername))
            else:
                os.makedirs(f'{new_save_directory}/AST/{foldername}', exist_ok=True)
                os.makedirs(f'{new_save_directory}/CFGAST/{foldername}', exist_ok=True)
                os.makedirs(f'{new_save_directory}/CFGDDGAST/{foldername}', exist_ok=True)
                os.makedirs(f'{new_save_directory}/ASTDDG/{foldername}', exist_ok=True)
            #print(f'Processing folder: {subfolders}')
            for filename in filenames:
                #print(f'Processing file: {filename}')
                file_path = os.path.join(foldername, filename)
                graph = None
                if graph_type == 'CFGAST':
                    graph = extract_cfg_ast(file_path)
                    save_graph_to_file(graph, f'{new_save_directory}/{foldername}', filename)
                elif graph_type == 'CFGDDGAST':
                    graph = extract_cfg_ddg_ast(filename)
                    save_graph_to_file(graph, f'{new_save_directory}/{foldername}', filename)
                elif graph_type == 'ASTDDG':
                    graph = extract_ast_ddg(file_path)
                    save_graph_to_file(graph, f'{new_save_directory}/{foldername}', filename)
                elif graph_type == 'AST':
                    graph = extract_ast(file_path)
                    save_graph_to_file(graph, f'{new_save_directory}/{foldername}', filename)
                elif graph_type == 'ALL':
                    ast = extract_ast(file_path)
                    ast_cfg = extract_cfg_ast(file_path)
                    ast_ddg = extract_ast_ddg(file_path)
                    cfg_ddg_ast = extract_cfg_ddg_ast(file_path)
                    save_graph_to_file(ast, f'{new_save_directory}/AST/{foldername}', filename)
                    save_graph_to_file(cfg_ddg_ast, f'{new_save_directory}/CFGDDGAST/{foldername}', filename)
                    save_graph_to_file(ast_ddg, f'{new_save_directory}/ASTDDG/{foldername}', filename)
                    save_graph_to_file(ast_cfg, f'{new_save_directory}/CFGAST/{foldername}', filename)
                    del ast, ast_cfg, cfg_ddg_ast, ast_ddg
                files_processed += 1
                del graph
                gc.collect()
                # os.system('cls' if os.name == 'nt' else 'clear') Doesn't work in the ide
                print(f'Processed {files_processed} files,   {filename} completed!')


if __name__ == '__main__':
    '''
    This analysis includes several different graph type analysis, this was done to verify that the graph edges were correct
    '''
    graph = extract_cfg_ddg_ast('paperbinary')
    visualize_graph_html(graph, 'figure.html')
