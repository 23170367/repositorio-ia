import java.util.List;
import java.util.PriorityQueue;

public class App{
    public static void main(String[] args) throws Exception{
        String initialState="7621 3458";
        String goalState = "12345678 ";

        SearchTree searchTree = new SearchTree(initialState, goalState);

        searchTree.deepFirstSearch();
        //searchTree.UniformCostSearch();
        //searchTree.breadthFirstSearch();
        //searchTree.deepLimitedSearch(82);
        //searchTree.deepIterativeSearch();

        System.out.println("End");

        /*System.out.println("INITIAL STATE> " + initialState);
        List<Node> children = NodeUtils.generateChildren(new Node(initialState, null));
        for(Node node : children){
            System.out.println(node.getState());
        }*/

        /*PriorityQueue<Node> queue = new PriorityQueue<Node>(new NodePriorityComparator());
        Node n1 = new Node("n1", null);
        n1.setCost(5);

        Node n2 = new Node("n2", null);
        n2.setCost(3);

        Node n3 = new Node("n3", null);
        n3.setCost(7);

        Node n4 = new Node("n4", null);
        n4.setCost(2);

        queue.add(n1); //5
        queue.add(n2); //3
        queue.add(n3); //7
        queue.add(n4); //2
        

        while(!queue.isEmpty()){
            Node node = queue.poll();
            System.out.println(node.getState() + " - " + node.getCost());
        }*/
    }
}   