#include "tensor.hpp"
namespace Ouroboros{
NDRange::NDRange(std::initializer_list<size_t> s,std::set<size_t> axis) : shape(s), axis(axis) {}

void NDRange::Iterator1::reset(std::size_t off) {
    offset = off;
    end_flag = false;
    for (std::size_t i = 0; i < index.size(); ++i) {
        index[i] = 0;
    }
}
NDRange::Iterator1::Iterator1(const std::vector<size_t>* shape_, const std::vector<size_t>* weight_, bool end,std::size_t off)
            : shape(shape_), weight(weight_), end_flag(end), offset(off){
            index.assign(shape->size(), 0);
            if (end_flag && !shape->empty())
                index[0] = (*shape)[0]; // make it “past-the-end”
}

NDRange::Iterator1& NDRange::Iterator1::operator++() {
    for (std::int64_t i = index.size() - 1; i >= 0; --i) {
        if (++index[i] < (*shape)[i]){
            offset += (*weight)[i];
            return *this;
        }
        offset -= (index[i]-1) * (*weight)[i];
        index[i] = 0;
    }
    end_flag = true;
    return *this;
}
bool NDRange::Iterator1::operator!=(const NDRange::Iterator1& other) const {
    if (end_flag != other.end_flag) return true;
    if (end_flag) return false;
    return index != other.index;
}


void NDRange::Range::reset(std::size_t off){
    it_start.reset(off);
    it_end.reset(off);
}
NDRange::Range::Range(const std::vector<size_t>* shape_, const std::vector<size_t>* weight_,std::size_t off)
            : it_start(shape_, weight_, false, off), it_end(shape_, weight_, true, off) {}

NDRange::Iterator1 NDRange::Range::begin() const{
    return it_start;
}
NDRange::Iterator1 NDRange::Range::end() const{
    return it_end;
}

NDRange::Iterator0::Iterator0(const Shape* shape_,const std::set<std::size_t>& axis, bool end)
            : end_flag(end){
    for (std::size_t i = 0; i < shape_->dim(); i++) {
        if (axis.find(i) != axis.end()) {
            strides0.push_back(shape_->get_stride(i));
            shape0.push_back(shape_->operator[](i));
        } else {
            strides1.push_back(shape_->get_stride(i));
            shape1.push_back(shape_->operator[](i));
        }
    }
    it0 = Iterator1(&shape0, &strides0, false);
    it1 = Range(&shape1, &strides1);
}

NDRange::Iterator0& NDRange::Iterator0::operator++() {
    ++it0;
    it1.reset(*it0);
    return *this;
}
bool NDRange::Iterator0::operator!=(const NDRange::Iterator0& other) const {
    if (end_flag != other.end_flag) return true;
    if (end_flag) return false;
    return it0 != other.it0;
}

NDRange::Iterator0 NDRange::begin() const { return Iterator0(&shape, axis, false); }
NDRange::Iterator0 NDRange::end()   const { return Iterator0(&shape, axis, true);  }
}