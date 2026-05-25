import java.util.*;

class iteratinglist {

    public static void main(String[] args) {

        List<Integer> list = new ArrayList<>();
        list.add(3);
        list.add(5);
        list.add(7);
        list.add(9);

        int max = Integer.MIN_VALUE;

        for (int num : list) {
            if (num > max) {
                max = num;
            }
        }

        System.out.println("List of Integers: " + list);
        System.out.println("Maximum element: " + max);
    }
}