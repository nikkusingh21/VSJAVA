public class Example {
    public static void main(String[] args) {
        StringBuilder sb = new StringBuilder("Parul");
        //paruluniversity
        sb.append(" ");
        sb.append("University");
        System.out.println(sb);
       sb.deleteCharAt(0);
       System.out.println(sb);
       sb.insert(0, "P");
       System.out.println(sb);
       sb.replace(7, 16,"UNIVERSITY");
       System.out.println(sb);
       sb.reverse();
       System.out.println(sb);
       sb.setCharAt(1, 'b');
       System.out.println(sb);
    }

}
//javap java.lang.StringBuilder