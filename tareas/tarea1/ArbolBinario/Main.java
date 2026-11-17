package tareas.tarea1.ArbolBinario;

public class Main {
    public static void main(String[] args) {
        Arbol arbol = new Arbol();

        arbol.insertar("Juan");
        arbol.insertar("Ana");
        arbol.insertar("Luis");

        if(arbol.vacio()) {
            System.out.println("El árbol está vacío.");
        } else {
            System.out.println("El árbol no está vacío.");
        }

        Nodo buscado = arbol.buscarNodo("Ana");

        if (buscado != null) {
            System.out.println("\nEncontrado: " + buscado.nombre);
        } else {
            System.out.println("\nNo encontrado");
        }
    }
}