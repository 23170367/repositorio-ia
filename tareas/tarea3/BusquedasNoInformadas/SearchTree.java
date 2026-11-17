package tareas.tarea3.BusquedasNoInformadas;

import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Set;
import java.util.Stack;

public class SearchTree{
    Node root;
    String initialState;
    String goalState;

    public SearchTree(String initialState, String goalState){
        this.initialState = initialState;
        this.goalState = goalState;
        this.root = new Node(initialState, null);
    }

    public void breadthFirstSearch() {
        int time = 0;
        // Crear estructura de datos para almacenar los nodos visitados
        Set<String> visited = new HashSet<String>();
        // 1. Buscar el nodo raíz y agregarlo a la cola
        Node currentNode = root;
        Queue<Node> queue = new LinkedList<>();
        queue.add(currentNode);
        // 2. Mientras la cola no esté vacía, hacer lo siguiente:
        while (!queue.isEmpty()) {
            time++;
            // 3. Sacar el primer nodo de la cola y verificar si es el nodo objetivo
            currentNode = queue.poll();
            visited.add(currentNode.getState());
            //System.out.println(NodeUtils.formatState(currentNode.getState()));
            if(currentNode.getState().equals(goalState)) {
                System.out.println("Goal state found: " + currentNode.getState());
                // print the path from root to goal
                printPath(currentNode);
                break;
            }
            // 4. Si no es el nodo objetivo, generar sus hijos y agregarlos a la cola
            List<Node> children = NodeUtils.generateChildren(currentNode);
            for (Node child : children) {
                if (!visited.contains(child.getState()))
                    queue.add(child);
            }
        }
        System.out.println("Time: " + time);
        System.out.println("Estados visitados: " + visited.size());
        System.out.println("Queue size: " + queue.size());
    }

    public void deepFirstSearch() {
        int time = 0;
        // Crear estructura de datos para almacenar los nodos visitados
        Set<String> visited = new HashSet<String>();
        // 1. Buscar el nodo raíz y agregarlo a la cola
        Node currentNode = root;
        Stack<Node> stack = new Stack<Node>(); //cambiar de Queue a Stack para Depth First Search
        stack.push(currentNode);
        // 2. Mientras la cola no esté vacía, hacer lo siguiente:
        while (!stack.isEmpty()) {
            time++;
            // 3. Sacar el primer nodo de la cola y verificar si es el nodo objetivo
            currentNode = stack.pop();
            visited.add(currentNode.getState());
            //System.out.println(NodeUtils.formatState(currentNode.getState()));
            if(currentNode.getState().equals(goalState)) {
                System.out.println("Goal state found:| " + currentNode.getState());
                // print the path from root to goal
                printPath(currentNode);
                break;
            }
            // 4. Si no es el nodo objetivo, generar sus hijos y agregarlos a la cola
            List<Node> children = NodeUtils.generateChildren(currentNode);
            for (Node child : children) {
                if (!visited.contains(child.getState()))
                    stack.add(child);
            }
        }
        System.out.println("Time: " + time);
        System.out.println("Estados visitados: " + visited.size());
        System.out.println("Stack size: " + stack.size());
    }

    public void UniformCostSearch() {
        int time = 0;
        // Crear estructura de datos para almacenar los nodos visitados
        Set<String> visited = new HashSet<String>();
        // 1. Buscar el nodo raíz y agregarlo a la cola
        Node currentNode = root;
        PriorityQueue<Node> queue = new PriorityQueue<>(new NodePriorityComparator());
        queue.add(currentNode);
        // 2. Mientras la cola no esté vacía, hacer lo siguiente:
        while (!queue.isEmpty()) {
            time++;
            // 3. Sacar el primer nodo de la cola y verificar si es el nodo objetivo
            currentNode = queue.poll();
            visited.add(currentNode.getState());
            //System.out.println(NodeUtils.formatState(currentNode.getState()));
            if(currentNode.getState().equals(goalState)) {
                System.out.println("Goal state found: " + currentNode.getState());
                // print the path from root to goal
                printPath(currentNode);
                break;
            }
            // 4. Si no es el nodo objetivo, generar sus hijos y agregarlos a la cola
            List<Node> children = NodeUtils.generateChildren(currentNode);
            for (Node child : children) {
                if (!visited.contains(child.getState())){
                    //Falto agregar la profundidad del hijo antes de calcular el costo
                    child.setDepth(child.getParent().getDepth() + 1);
                    child.setCost(child.getParent().getDepth() + 1); // set the cost of the child node
                    queue.add(child);
                }
            }
        }
        System.out.println("Time: " + time);
        System.out.println("Estados visitados: " + visited.size());
        System.out.println("Queue size: " + queue.size());
    }

    public boolean deepLimitedSearch(int limit) {
        int time = 0;
        // Crear estructura de datos para almacenar los nodos visitados
        Set<String> visited = new HashSet<String>();
        // 1. Buscar el nodo raíz y agregarlo a la pila
        Node currentNode = root;
        Stack<Node> stack = new Stack<Node>();
        stack.push(currentNode);
        // 2. Mientras la pila no esté vacía, hacer lo siguiente:
        while (!stack.isEmpty()) {
            time++;
            // 3. Sacar el primer nodo de la pila y verificar si es el nodo objetivo
            currentNode = stack.pop();
            visited.add(currentNode.getState());
            //System.out.println(NodeUtils.formatState(currentNode.getState()));
            if(currentNode.getState().equals(goalState)) {
                System.out.println("Goal state found at limit " + limit + ": " + currentNode.getState());
                // print the path from root to goal
                printPath(currentNode);
                System.out.println("Time: " + time);
                System.out.println("Estados visitados: " + visited.size());
                System.out.println("Stack size: " + stack.size());
                return true; // Retorna true porque ya encontró la meta
            }
            // 4. Si no es el nodo objetivo y no supera el límite, generar sus hijos y agregarlos a la pila
            if (currentNode.getDepth() < limit) {
                List<Node> children = NodeUtils.generateChildren(currentNode);
                for (Node child : children) {
                    if (!visited.contains(child.getState())) {
                        child.setDepth(child.getParent().getDepth() + 1);
                        stack.add(child);
                    }
                }
            }
        }
        System.out.println("Time: " + time);
        System.out.println("Estados visitados: " + visited.size());
        System.out.println("Stack size: " + stack.size());
        return false; // No encontró la meta con este límite
    }

    public void deepIterativeSearch() {
        int limit = 0;
        // Combinar con búsqueda limitada: incrementar el límite hasta que depthLimitedSearch devuelva true
        while (!deepLimitedSearch(limit)) {
            limit++;
        }
    }

    private int heuristic(String state) {
        int heuristic = 0;
        for (int i = 0; i < state.length(); i++) {
            char c = state.charAt(i);
            if (c != ' ') {
                int goalIndex = goalState.indexOf(c);
                int currentRow = i / 3;
                int currentCol = i % 3;
                int goalRow = goalIndex / 3;
                int goalCol = goalIndex % 3;
                heuristic += Math.abs(currentRow - goalRow) + Math.abs(currentCol - goalCol);
            }
        }
        return heuristic;
    }

    private void printPath(Node node) {
        if (node == null) {
            return;
        }
        printPath(node.getParent());
        System.out.println(NodeUtils.formatState(node.getState()));
    }
}