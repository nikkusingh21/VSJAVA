class Student {
    int rollNo;
    String name;
    double marks;
    char grade;
    boolean result;
}

public class DatatypeRoll {
    public static void main(String[] args) {

        Student[] s = new Student[5];

        for (int i = 0; i < 5; i++) {
            s[i] = new Student();
        }

        s[0].rollNo = 101;
        s[0].name = "Rahul";
        s[0].marks = 85.5;
        s[0].grade = 'A';
        s[0].result = true;

        s[1].rollNo = 102;
        s[1].name = "Aman";
        s[1].marks = 78.0;
        s[1].grade = 'B';
        s[1].result = true;

        s[2].rollNo = 103;
        s[2].name = "Niku";
        s[2].marks = 92.3;
        s[2].grade = 'A';
        s[2].result = true;

        s[3].rollNo = 104;
        s[3].name = "Priya";
        s[3].marks = 66.5;
        s[3].grade = 'C';
        s[3].result = true;

        for (int i = 0; i < 5; i++) {
            System.out.println("Roll No: " + s[i].rollNo);
            System.out.println("Name: " + s[i].name);
            System.out.println("Marks: " + s[i].marks);
            System.out.println("Grade: " + s[i].grade);
            System.out.println("Result: " + s[i].result);
            System.out.println();
        }
    }
}