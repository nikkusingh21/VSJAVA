class Linkedlist {
    public static class Node{
        int data;
        Node next;
        //constructor
        Node(int val){
            this.data = val;
            this.next = null;
        }
    }
    public static void main(String[] args) {
        Node n1 = new Node(5);
        Node n2 = new Node(6);
        Node n3 = new Node(7);
        Node n4 = new Node(8);
        //connection
        n1.next = n2;
        n2.next = n3;
        n3.next = n4;
        //traversal
        Node temp = n1;
        while(temp != null){
            System.out.print(temp.data + "->");
            temp = temp.next;
        }
        System.out.println("null");
    }
}
