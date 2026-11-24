#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace resectionlab::wall_sweep {
using Vec = std::array<double, 3>;
using Tri = std::array<Vec, 3>;
using Ids = std::array<int64_t, 3>;
constexpr double clearance = 1e-9;
constexpr int max_depth = 12;
constexpr int64_t max_visits = 4'000'000;
constexpr double max_coordinate = 4096.;
constexpr size_t max_items = 500'000;

Vec add(const Vec& a, const Vec& b) { return {a[0]+b[0], a[1]+b[1], a[2]+b[2]}; }
Vec sub(const Vec& a, const Vec& b) { return {a[0]-b[0], a[1]-b[1], a[2]-b[2]}; }
Vec mid(const Vec& a, const Vec& b) {
    return {(.5*a[0])+(.5*b[0]), (.5*a[1])+(.5*b[1]), (.5*a[2])+(.5*b[2])};
}
Vec cross(const Vec& a, const Vec& b) {
    return {a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]};
}
double dot(const Vec& a, const Vec& b) { return (a[0]*b[0]+a[1]*b[1])+a[2]*b[2]; }
Vec load(const double* data) { return {data[0], data[1], data[2]}; }
bool bounded(const Vec& v) {
    return std::all_of(v.begin(), v.end(), [](double x) { return std::abs(x) <= max_coordinate; });
}

struct Box { Vec lo, hi; };
Box merged(const Box& a, const Box& b) {
    Box box;
    for (int d=0; d<3; ++d) {
        box.lo[d] = std::min(a.lo[d], b.lo[d]);
        box.hi[d] = std::max(a.hi[d], b.hi[d]);
    }
    return box;
}
Box bounds(const Tri& tri) {
    Box box{tri[0], tri[0]};
    for (int i=1; i<3; ++i) box = merged(box, {tri[i], tri[i]});
    return box;
}
bool overlaps(const Box& a, const Box& b) {
    for (int d=0; d<3; ++d) {
        if (a.lo[d] > b.hi[d]+clearance || b.lo[d] > a.hi[d]+clearance) return false;
    }
    return true;
}

Tri midpoint(const Tri& a, const Tri& b) {
    return {mid(a[0], b[0]), mid(a[1], b[1]), mid(a[2], b[2])};
}

// Translation-invariant projections. One axis and sign must work for both
// endpoints. All nine signed vertex-pair differences are then positive
// throughout the linear interval, even if the projected ranges drift together.
struct Sample {
    Tri first, second;
    explicit Sample(const Tri& a, const Tri& b) {
        for (int i=0; i<3; ++i) {
            first[i]=sub(a[i], a[0]);
            second[i]=sub(b[i], a[0]);
        }
    }
    int side(const Vec& axis, double margin) const {
        const double a=dot(first[1],axis), b=dot(first[2],axis);
        const double x=dot(second[0],axis), y=dot(second[1],axis), z=dot(second[2],axis);
        if (std::max({0.,a,b})+margin < std::min({x,y,z})) return 1;
        if (std::max({x,y,z})+margin < std::min({0.,a,b})) return -1;
        return 0;
    }
};

template <typename Predicate>
bool separating_axis(const Tri& a, const Tri& b, Predicate separates) {
    for (const Vec& axis : {Vec{1.,0.,0.}, Vec{0.,1.,0.}, Vec{0.,0.,1.}}) {
        if (separates(axis, clearance)) return true;
    }
    Tri ae, be;
    for (int i=0; i<3; ++i) {
        ae[i]=sub(a[(i+1)%3],a[i]);
        be[i]=sub(b[(i+1)%3],b[i]);
    }
    const Vec an=cross(ae[0],ae[1]), bn=cross(be[0],be[1]);
    auto test = [&](const Vec& axis) {
        const double squared = dot(axis,axis);
        return squared >= 1e-30 && separates(axis, clearance*std::sqrt(squared));
    };
    if (test(an) || test(bn)) return true;
    for (const Vec& u: ae) for (const Vec& v: be) if (test(cross(u,v))) return true;
    for (const Vec& u: ae) if (test(cross(an,u))) return true;
    for (const Vec& v: be) if (test(cross(bn,v))) return true;
    return false;
}

bool instantaneous(const Tri& a, const Tri& b) {
    const Sample sample(a,b);
    return !separating_axis(a,b,[&](const Vec& axis, double margin) {
        return sample.side(axis,margin) != 0;
    });
}

struct Stroke { Tri prior, next; };
bool clear_interval(const Stroke& first, const Stroke& second, int depth, int64_t& intervals) {
    if (++intervals > max_visits) return false;
    const Tri first_mid=midpoint(first.prior,first.next), second_mid=midpoint(second.prior,second.next);
    const Sample start(first.prior,second.prior), end(first.next,second.next);
    if (separating_axis(first_mid,second_mid,[&](const Vec& axis, double margin) {
        const int side=start.side(axis,margin);
        return side && side==end.side(axis,margin);
    })) return true;
    if (depth==max_depth || instantaneous(first_mid,second_mid)) return false;
    return clear_interval({first.prior,first_mid},{second.prior,second_mid},depth+1,intervals)
        && clear_interval({first_mid,first.next},{second_mid,second.next},depth+1,intervals);
}

using Rays = std::array<Vec, 4>;

Rays vertex_rays(const Tri& a, const Tri& b, int corner_a, int corner_b) {
    return {sub(a[(corner_a+1)%3], a[corner_a]), sub(a[(corner_a+2)%3], a[corner_a]),
            sub(b[corner_b], b[(corner_b+1)%3]), sub(b[corner_b], b[(corner_b+2)%3])};
}

bool positive_rays(const Rays& rays, const Vec& axis, double margin) {
    return std::all_of(rays.begin(), rays.end(), [&](const Vec& ray) { return dot(ray,axis)>margin; });
}

template <typename Predicate>
bool vertex_axis(Rays rays, Predicate separates) {
    auto test = [&](const Vec& axis) {
        const double squared=dot(axis,axis);
        return squared>=1e-30 && separates(axis,clearance*std::sqrt(squared));
    };
    Vec total{};
    for (Vec& ray: rays) {
        const double length=std::sqrt(dot(ray,ray));
        if (length<=clearance) return false;
        for (double& coordinate: ray) coordinate/=length;
        total=add(total,ray);
    }
    if (test(total)) return true;
    // A separating axis can be chosen as the closest point of the unit-ray
    // convex hull to the origin. Its supporting feature is a vertex, edge,
    // or triangle. Equal-length unit rays put edge projections at midpoints.
    for (const Vec& ray: rays) if (test(ray)) return true;
    for (int i=0; i<4; ++i) for (int j=i+1; j<4; ++j) {
        if (test(add(rays[i],rays[j]))) return true;
        for (int k=j+1; k<4; ++k) {
            const Vec normal=cross(sub(rays[j],rays[i]), sub(rays[k],rays[i]));
            if (test(normal) || test(sub(Vec{},normal))) return true;
        }
    }
    return false;
}

bool vertex_interval(const Rays& before, const Rays& after, int depth, int64_t& intervals) {
    if (++intervals>max_visits) return false;
    Rays middle;
    for (int i=0; i<4; ++i) middle[i]=mid(before[i],after[i]);
    bool middle_clear=false;
    if (vertex_axis(middle,[&](const Vec& axis, double margin) {
        if (!positive_rays(middle,axis,margin)) return false;
        middle_clear=true;
        // The signed rays are linear in time. Strict endpoint half-spaces
        // let the faces touch at their common vertex only for this interval.
        return positive_rays(before,axis,margin) && positive_rays(after,axis,margin);
    })) return true;
    return middle_clear && depth<max_depth
        && vertex_interval(before,middle,depth+1,intervals)
        && vertex_interval(middle,after,depth+1,intervals);
}

struct Face { Tri rest; Ids vertices, owners; Box box; };
struct Node {
    int begin, end, lhs=-1, rhs=-1;
};

struct Wall {
    std::vector<Face> faces;
    std::vector<int> order;
    std::vector<Node> nodes;
    size_t owner_count;
    bool check_shared_vertex=false;

    int build(int begin, int end) {
        const int index=int(nodes.size());
        Box box=faces[order[begin]].box;
        for (int j=begin+1; j<end; ++j) box=merged(box,faces[order[j]].box);
        nodes.push_back({begin,end});
        if (end-begin <= 8) return index;
        const Vec extent=sub(box.hi,box.lo);
        const int axis=int(std::max_element(extent.begin(),extent.end())-extent.begin());
        const int split=(begin+end)/2;
        std::nth_element(order.begin()+begin,order.begin()+split,order.begin()+end,[&](int a,int b) {
            const auto& x=faces[a].box;
            const auto& y=faces[b].box;
            const double xc=x.lo[axis]+x.hi[axis], yc=y.lo[axis]+y.hi[axis];
            return xc==yc ? a<b : xc<yc;
        });
        const int left=build(begin,split), right=build(split,end);
        nodes[index].lhs=left;
        nodes[index].rhs=right;
        return index;
    }
};

bool adjacent(const Face& a, const Face& b) {
    for (int64_t id: a.vertices) for (int64_t other: b.vertices) if (id==other) return true;
    return false;
}

int shared_corner(const Face& a, const Face& b) {
    int corner=-1;
    for (int i=0; i<3; ++i) for (int j=0; j<3; ++j) if (a.vertices[i]==b.vertices[j]) {
        if (corner>=0) return -2;
        corner=3*i+j;
    }
    return corner;
}

bool same_increment(const Face& a, const Face& b, const std::vector<Vec>& increments) {
    const Vec& first=increments[a.owners[0]];
    for (int j=0; j<3; ++j) {
        if (increments[a.owners[j]] != first || increments[b.owners[j]] != first) return false;
    }
    return true;
}

struct Check {
    const Wall& wall;
    std::vector<Stroke> positions;
    std::vector<Vec> increments;
    std::vector<uint8_t> dirty, node_dirty;
    std::vector<Box> swept, node_bounds;
    int64_t visits=0, pairs=0, intervals=0;
    bool static_check;

    Check(const Wall& wall_, const double* prior, const double* next, bool audit)
        : wall(wall_), positions(wall.faces.size()), increments(wall.owner_count), dirty(wall.faces.size()),
          node_dirty(wall.nodes.size()), swept(wall.faces.size()), node_bounds(wall.nodes.size()),
          static_check(audit) {
        for (size_t i=0; i<increments.size(); ++i) increments[i]=sub(load(next+3*i),load(prior+3*i));
        for (size_t i=0; i<wall.faces.size(); ++i) {
            const Face& face=wall.faces[i];
            Stroke& path=positions[i];
            for (int j=0; j<3; ++j) {
                const size_t owner=face.owners[j];
                path.prior[j]=add(face.rest[j], load(prior+3*owner));
                path.next[j]=add(face.rest[j], load(next+3*owner));
                dirty[i] |= increments[owner]!=Vec{};
            }
            dirty[i] |= static_check;
            swept[i]=merged(bounds(path.prior),bounds(path.next));
        }
        for (int index=int(wall.nodes.size())-1; index>=0; --index) {
            const Node& node=wall.nodes[index];
            if (node.lhs>=0) {
                node_bounds[index]=merged(node_bounds[node.lhs],node_bounds[node.rhs]);
                node_dirty[index]=node_dirty[node.lhs] || node_dirty[node.rhs];
                continue;
            }
            Box box=swept[wall.order[node.begin]];
            for (int j=node.begin; j<node.end; ++j) {
                const int face=wall.order[j];
                box=merged(box,swept[face]);
                node_dirty[index] |= dirty[face];
            }
            node_bounds[index]=box;
        }
    }

    bool pair_clear(int a, int b) {
        if ((!dirty[a] && !dirty[b]) || !overlaps(swept[a],swept[b])) return true;
        const auto& x=wall.faces[a];
        const auto& y=wall.faces[b];
        if (!static_check && same_increment(x,y,increments)) return true;
        if (!wall.check_shared_vertex && adjacent(x,y)) return true;
        const int corner=wall.check_shared_vertex ? shared_corner(x,y) : -1;
        // The local hinge guard checks faces sharing an edge.
        if (corner==-2) return true;
        if (++pairs > max_visits) return false;
        const auto& first=positions[a];
        const auto& second=positions[b];
        if (corner>=0) {
            const auto after=vertex_rays(first.next,second.next,corner/3,corner%3);
            if (static_check) return vertex_axis(after,[&](const Vec& axis, double margin) {
                return positive_rays(after,axis,margin);
            });
            const auto before=vertex_rays(first.prior,second.prior,corner/3,corner%3);
            return vertex_interval(before,after,0,intervals);
        }
        if (static_check) return !instantaneous(first.next,second.next);
        return clear_interval(first,second,0,intervals);
    }

    bool clear_nodes(int a, int b) {
        if ((!node_dirty[a] && !node_dirty[b]) || !overlaps(node_bounds[a],node_bounds[b])) return true;
        if (++visits > max_visits) return false;
        const Node& x=wall.nodes[a];
        const Node& y=wall.nodes[b];
        if (a==b && x.lhs>=0) {
            return clear_nodes(x.lhs,x.lhs) && clear_nodes(x.lhs,x.rhs) && clear_nodes(x.rhs,x.rhs);
        }
        if (x.lhs>=0 && (y.lhs<0 || x.end-x.begin>=y.end-y.begin)) {
            return clear_nodes(x.lhs,b) && clear_nodes(x.rhs,b);
        }
        if (y.lhs>=0) return clear_nodes(a,y.lhs) && clear_nodes(a,y.rhs);
        for (int i=x.begin; i<x.end; ++i) {
            for (int j=a==b?i+1:y.begin; j<y.end; ++j) {
                if (!pair_clear(wall.order[i],wall.order[j])) return false;
            }
        }
        return true;
    }

    bool run() {
        return wall.nodes.empty() || clear_nodes(0,0);
    }
};
} // namespace resectionlab::wall_sweep
