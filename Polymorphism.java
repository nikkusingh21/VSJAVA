class Car {
    void drive() {
        System.out.println("Car is driving");
    }
}

class SportsCar extends Car {
    void drive() {
        System.out.println("SportsCar is driving fast");
    }
}

class PolymorphismDemo {
    public static void main(String[] args) {
        Car myCar = new SportsCar(); 
        myCar.drive(); 
    }
}