class Ride(object):
    def __init__(self, id, start_pos, end_pos, start_time, end_time):
        self.id = id
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.start_time = start_time
        self.end_time = end_time
        self.length = distance(self.start_pos, self.end_pos)

    def __str__(self):
        return "{} : {} {} - {} {} - {} - {}".format(
            self.id,
            self.start_pos[0], self.start_pos[1],
            self.end_pos[0], self.end_pos[1],
            self.start_time, self.end_time
        )


class Vehicle(object):
    def __init__(self, id):
        self.id = id
        self.pos = (0, 0)
        self.free_time = 0
        self.rides = []


def distance(start, end):
    return abs(end[1] - start[1]) + abs(end[0] - start[0])


def output(vehicles, file_name):
    with open(file_name, 'w') as file:
        for vehicle in vehicles:
            num_of_rides = len(vehicle.rides)
            rides_ids = [str(ride.id) for ride in vehicle.rides]
            print(num_of_rides, ' '.join(rides_ids))
            rides_str = ' '.join(rides_ids)
            file.write(str(num_of_rides) + " " + rides_str + '\n')


paths = [
    'a_example.in',
    'b_should_be_easy.in',
    'c_no_hurry.in',
    'd_metropolis.in',
    'e_high_bonus.in'
]
# paths = [
#     'a_example.in'
# ]
for path in paths:

    step_cnt = 0
    vehicles = []
    rides = []
    with open(path, 'r') as file:
        info = file.readline().split()
        rows, cols, vehicles_num, rides_num, bonus, steps = map(int, info)
        for i in range(rides_num):
            info = file.readline().split()
            start_x, start_y, end_x, end_y, s_time, e_time = map(int, info)
            rides.append(Ride(i, (start_x, start_y), (end_x, end_y), s_time, e_time))
    for i in range(vehicles_num):
        vehicles.append(Vehicle(i))

    ride_vehicle = dict()
    for vehicle in vehicles:
        ride_vehicle[vehicle.id] = []

    rides.sort(key=lambda x: distance(x.start_pos, (0, 0)))
    for ride in rides:
        minimum = 20001
        min_bonus = 0
        best_vehicle = -1
        best_vehicle_bonus = -1
        best_ride_time_bonus = -1
        best_ride_time = -1
        for vehicle in vehicles:
            start = distance(vehicle.pos, ride.start_pos)
            ride_time = max(
                start + vehicle.free_time, ride.start_time
            ) + ride.length
            maybe_bonus = start + vehicle.free_time - ride.start_time
            match = ride_time < ride.end_time

            if match:
                if maybe_bonus <= 0:
                    min_bonus = maybe_bonus
                    best_vehicle_bonus = vehicle
                    best_ride_time_bonus = ride_time
                elif start < minimum:
                    minimum = start
                    best_vehicle = vehicle
                    best_ride_time = ride_time
        if best_vehicle_bonus != -1:
            best_vehicle = best_vehicle_bonus
            best_ride_time = best_ride_time_bonus
        if best_vehicle != -1:
            best_vehicle.rides.append(ride)
            ride_vehicle[best_vehicle.id].append(ride.id)
            best_vehicle.pos = ride.end_pos
            best_vehicle.free_time = best_ride_time

    output(vehicles, path[0])
    print()
    print()
