class CountSumSortedLinked {

    public static class Node {
        int data;
        Node next;

        // constructor
        Node(int val) {
            this.data = val;
            this.next = null;
        }
    }

    public static void main(String[] args) {

        Node n1 = new Node(5);
        Node n2 = new Node(6);
        Node n3 = new Node(7);
        Node n4 = new Node(8);

        // connection
        n1.next = n2;
        n2.next = n3;
        n3.next = n4;

        // traversal + count + sum + sorted check
        Node temp = n1;

        int count = 0;
        int sum = 0;
        boolean sorted = true;

        while (temp != null) {

            System.out.print(temp.data + " -> ");

            count++;
            sum += temp.data;

            // sorted check
            if (temp.next != null && temp.data > temp.next.data) {
                sorted = false;
            }

            temp = temp.next;
        }

        System.out.println("null");

        System.out.println("Count = " + count);
        System.out.println("Sum = " + sum);

        if (sorted) {
            System.out.println("Linked List is Sorted");
        } else {
            System.out.println("Linked List is Not Sorted");
        }
    }
}