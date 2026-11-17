package tareas.tarea1;

public class Arbol {
    Nodo raiz;

    public boolean vacio() {
        return raiz == null;
    }

    public Nodo buscarNodo(String nombre) {
        Nodo actual = raiz;

        while (actual != null) {
            int comparacion = nombre.compareToIgnoreCase(actual.nombre);

            if (comparacion == 0) {
                return actual;
            } else if (comparacion < 0) {
                actual = actual.izquierdo;
            } else {
                actual = actual.derecho;
            }
        }
        return null;
    }

    public void insertar(String nombre) {
        if (raiz == null) {
            raiz = new Nodo(nombre);
            return;
        }

        Nodo actual = raiz;
        while (true) {
            int comparacion = nombre.compareToIgnoreCase(actual.nombre);

            if (comparacion < 0) {
                if (actual.izquierdo == null) {
                    actual.izquierdo = new Nodo(nombre);
                    return;
                }
                actual = actual.izquierdo;
            } else if (comparacion > 0) {
                if (actual.derecho == null) {
                    actual.derecho = new Nodo(nombre);
                    return;
                }
                actual = actual.derecho;
            } else return;
        }
    }
}